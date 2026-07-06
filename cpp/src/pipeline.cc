#include "pyplus/pipeline.hh"

#include <stdexcept>
#include <utility>

#include "pyplus/features.hh"

namespace pyplus {

AutomatonResult segment_automaton(const std::string &image_path,
                                  const KNNClassifier *classifier) {
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
        const std::vector<float> features = compute_hog(label.crop);
        label.symbol = classifier->predict(features);
      }
    }
  }

  AutomatonResult result;
  result.states = std::move(states);
  result.matrix = std::move(adjacency.matrix);
  result.arrows = std::move(adjacency.arrows);
  result.initial = adjacency.initial;
  return result;
}

} // namespace pyplus
