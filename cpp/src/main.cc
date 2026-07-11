// Point d'entree du portage C++ (equivalent de python/src/main.py) :
// segmentation + reconnaissance d'une image d'automate.
//
//     pyplus <image> [--letters data/knn_letters.bin]
//            [--digits data/knn_digits.bin] [--table sortie.txt]
//     pyplus --batch <corpus> --out <dossier> [--letters ...] [--digits ...]
//
// En mode batch, chaque PNG du corpus (recursif) produit une table .txt
// dans le dossier de sortie, en miroir de l'arborescence ; les modeles ne
// sont charges qu'une fois.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/export_table.hh"
#include "pyplus/model_io.hh"
#include "pyplus/pipeline.hh"

namespace {

// Affiche un resume textuel de la segmentation (cf. print_summary Python).
void print_summary(const pyplus::AutomatonResult &result) {
  const std::vector<pyplus::State> &states = result.states;

  std::vector<std::string> names(states.size());
  for (std::size_t i = 0; i < states.size(); ++i)
    names[i] =
        states[i].name.empty() ? "s" + std::to_string(i) : states[i].name;

  std::cout << states.size() << " etat(s) :\n";
  for (std::size_t i = 0; i < states.size(); ++i) {
    std::string tag;
    if (result.initial && *result.initial == static_cast<int>(i))
      tag += "initial";
    if (states[i].accepting)
      tag += (tag.empty() ? "" : ", ") + std::string("acceptant");
    std::cout << "  " << names[i] << ": centre=("
              << std::lround(states[i].center_x) << ", "
              << std::lround(states[i].center_y) << ")"
              << (tag.empty() ? "" : " [" + tag + "]") << "\n";
  }

  std::cout << "\n" << result.arrows.size() << " transition(s) detectee(s) :\n";
  for (const pyplus::Arrow &arrow : result.arrows) {
    std::string symbols;
    std::size_t n_labels = 0;
    for (const pyplus::ArrowLabel &label : arrow.labels) {
      ++n_labels;
      if (label.symbol)
        symbols += (symbols.empty() ? "" : ", ") + *label.symbol;
    }
    if (!symbols.empty())
      std::cout << "  " << names[arrow.source] << " --" << symbols << "--> "
                << names[arrow.dest] << "\n";
    else
      std::cout << "  " << names[arrow.source] << " -> " << names[arrow.dest]
                << "  (" << n_labels << " etiquette(s) non reconnue(s))\n";
  }

  std::size_t width = 2;
  for (const std::string &name : names)
    width = std::max(width, name.size());
  std::cout << "\nMatrice d'adjacence :\n" << std::string(width + 3, ' ');
  for (std::size_t j = 0; j < states.size(); ++j)
    std::cout << (j > 0 ? "  " : "")
              << std::string(width - names[j].size(), ' ') << names[j];
  std::cout << "\n";
  for (std::size_t i = 0; i < states.size(); ++i) {
    std::cout << std::string(width - names[i].size(), ' ') << names[i] << ":  ";
    for (std::size_t j = 0; j < states.size(); ++j)
      std::cout << (j > 0 ? "   " : "")
                << (result.matrix[i][j] >= 0 ? "1" : "0");
    std::cout << "\n";
  }
}

int run_batch(const std::string &corpus_dir, const std::string &out_dir,
              const pyplus::KNNClassifier *classifier,
              const pyplus::KNNClassifier *digit_classifier) {
  namespace fs = std::filesystem;

  std::vector<fs::path> images;
  for (const auto &entry : fs::recursive_directory_iterator(corpus_dir))
    if (entry.path().extension() == ".png")
      images.push_back(entry.path());
  std::sort(images.begin(), images.end());
  if (images.empty()) {
    std::cerr << "aucun PNG dans " << corpus_dir << "\n";
    return 1;
  }

  const auto start = std::chrono::steady_clock::now();
  int n_errors = 0;
  for (const fs::path &png : images) {
    fs::path out = fs::path(out_dir) / fs::relative(png, corpus_dir);
    out.replace_extension(".txt");
    fs::create_directories(out.parent_path());
    try {
      pyplus::save_table(
          pyplus::segment_automaton(png.string(), classifier, digit_classifier),
          out.string());
    } catch (const std::exception &e) {
      std::cerr << "[!] " << png.string() << " : " << e.what() << "\n";
      ++n_errors;
    }
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  std::cout << images.size() - n_errors << "/" << images.size()
            << " tables ecrites dans " << out_dir << " (" << seconds << "s)\n";
  return n_errors > 0 ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string image_path;
  std::string letters_path = "data/knn_letters.bin";
  std::string digits_path = "data/knn_digits.bin";
  std::string table_path;
  std::string batch_dir;
  std::string out_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--letters" && i + 1 < argc)
      letters_path = argv[++i];
    else if (arg == "--digits" && i + 1 < argc)
      digits_path = argv[++i];
    else if (arg == "--table" && i + 1 < argc)
      table_path = argv[++i];
    else if (arg == "--batch" && i + 1 < argc)
      batch_dir = argv[++i];
    else if (arg == "--out" && i + 1 < argc)
      out_dir = argv[++i];
    else if (image_path.empty())
      image_path = arg;
  }
  const bool batch = !batch_dir.empty();
  if ((batch && out_dir.empty()) || (!batch && image_path.empty())) {
    std::cerr << "usage: " << argv[0]
              << " <image> [--letters data/knn_letters.bin]"
                 " [--digits data/knn_digits.bin] [--table sortie.txt]\n"
              << "       " << argv[0]
              << " --batch <corpus> --out <dossier>"
                 " [--letters ...] [--digits ...]\n";
    return 1;
  }

  pyplus::KNNClassifier classifier(3, true);
  bool has_classifier = true;
  try {
    classifier.fit(pyplus::load_model(letters_path));
  } catch (const std::exception &e) {
    has_classifier = false;
    std::cerr << "[!] " << e.what()
              << " : segmentation seule, sans reconnaissance des "
                 "symboles.\n";
  }
  pyplus::KNNClassifier digit_classifier(3, true);
  bool has_digits = true;
  try {
    digit_classifier.fit(pyplus::load_model(digits_path));
  } catch (const std::exception &e) {
    has_digits = false;
    std::cerr << "[!] " << e.what() << " : noms d'etats non reconnus.\n";
  }

  try {
    if (batch)
      return run_batch(batch_dir, out_dir,
                       has_classifier ? &classifier : nullptr,
                       has_digits ? &digit_classifier : nullptr);

    const pyplus::AutomatonResult result = pyplus::segment_automaton(
        image_path, has_classifier ? &classifier : nullptr,
        has_digits ? &digit_classifier : nullptr);
    print_summary(result);
    if (!table_path.empty()) {
      pyplus::save_table(result, table_path);
      std::cout << "\nTable ecrite dans " << table_path << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}
