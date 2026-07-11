#include "pyplus/pipeline.hh"

#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "pyplus/features.hh"

namespace pyplus {

namespace {

// Miroir de _NAME_PATTERN (re.fullmatch(r"s[0-9]+")) du prototype.
bool is_valid_state_name(const std::string &name) {
  if (name.size() < 2 || name[0] != 's')
    return false;
  for (std::size_t i = 1; i < name.size(); ++i)
    if (name[i] < '0' || name[i] > '9')
      return false;
  return true;
}

// Lit le nom d'un etat depuis son crop : premier caractere via le modele
// lettres (le 's'), les suivants via le modele chiffres. Chaine vide si le
// crop n'a pas au moins 2 caracteres ou si une prediction est rejetee.
std::string recognize_state_name(const cv::Mat &name_crop,
                                 const KNNClassifier &letters,
                                 const KNNClassifier &digits) {
  const std::vector<cv::Mat> crops = segment_name_characters(name_crop);
  if (crops.size() < 2)
    return "";
  // normalize_crop recadre au ras du trait, comme a l'entrainement
  std::string name;
  const std::optional<std::string> first =
      letters.predict(compute_hog(normalize_crop(crops[0])));
  if (!first)
    return "";
  name += *first;
  for (std::size_t i = 1; i < crops.size(); ++i) {
    const std::optional<std::string> digit =
        digits.predict(compute_hog(normalize_crop(crops[i])));
    if (!digit)
      return "";
    name += *digit;
  }
  return name;
}

// Attribue son nom final a chaque etat (State::name), miroir exact de
// assign_state_names du prototype : un nom brut est retenu s'il est valide
// (motif s[0-9]+) et unique parmi les noms bruts ; sinon l'etat retombe sur
// le plus petit nom d'indice s{i} encore libre, dans l'ordre des etats.
void assign_state_names(std::vector<State> &states,
                        const std::vector<std::string> &raw_names) {
  std::vector<std::string> valid(states.size());
  std::map<std::string, int> counts;
  for (std::size_t i = 0; i < states.size(); ++i) {
    if (is_valid_state_name(raw_names[i])) {
      valid[i] = raw_names[i];
      ++counts[valid[i]];
    }
  }

  std::set<std::string> used;
  for (std::size_t i = 0; i < states.size(); ++i) {
    if (!valid[i].empty() && counts[valid[i]] == 1) {
      states[i].name = valid[i];
      used.insert(valid[i]);
    } else {
      states[i].name.clear();
    }
  }

  int next_index = 0;
  for (State &state : states) {
    if (!state.name.empty())
      continue;
    while (used.count("s" + std::to_string(next_index)))
      ++next_index;
    state.name = "s" + std::to_string(next_index);
    used.insert(state.name);
  }
}

} // namespace

AutomatonResult segment_automaton(const std::string &image_path,
                                  const KNNClassifier *classifier,
                                  const KNNClassifier *digit_classifier) {
  // 1. chargement + binarisation (trait=255, fond=0)
  const cv::Mat gray = load_image(image_path);
  if (gray.empty())
    throw std::runtime_error("Image introuvable : " + image_path);
  const cv::Mat binary = to_binary(gray);

  std::vector<State> states = segment_states(binary);

  // 2. cercles residuels = deuxieme cercle des etats acceptants
  const cv::Mat img_without_inner = remove_states_from_img(binary, states);
  const std::vector<State> residual_circles = segment_states(img_without_inner);
  merge_states_and_acceptants(states, residual_circles);
  const cv::Mat img_clean = remove_states_from_img(binary, states);

  // 3. tetes de fleche -> arcs (suivi de trace) -> matrice d'adjacence
  const std::vector<Tip> tips = find_triangles(img_clean);
  AdjacencyResult adjacency = build_adjacency_matrix(states, tips, img_clean);

  // 4. etiquettes : isolation puis rattachement a l'arc le plus proche
  const cv::Mat img_labels = isolate_labels(img_clean, tips, adjacency.arrows);
  assign_labels_to_arrows(img_labels, adjacency.arrows);

  // 5. reconnaissance des symboles
  if (classifier != nullptr) {
    for (Arrow &edge : adjacency.arrows) {
      for (ArrowLabel &label : edge.labels) {
        // normalize_crop recadre au ras du trait : meme cadrage que les
        // glyphes d'entrainement (crop_to_content)
        const std::vector<float> features =
            compute_hog(normalize_crop(label.crop));
        label.symbol = classifier->predict(features);
      }
    }
  }

  // 6. noms d'etats : lus dans l'image si les deux modeles sont presents,
  //    repli s{i} sinon
  std::vector<std::string> raw_names(states.size());
  if (classifier != nullptr && digit_classifier != nullptr)
    for (std::size_t i = 0; i < states.size(); ++i)
      raw_names[i] = recognize_state_name(states[i].name_crop, *classifier,
                                          *digit_classifier);
  assign_state_names(states, raw_names);

  AutomatonResult result;
  result.states = std::move(states);
  result.matrix = std::move(adjacency.matrix);
  result.arrows = std::move(adjacency.arrows);
  result.initial = adjacency.initial;
  return result;
}

} // namespace pyplus
