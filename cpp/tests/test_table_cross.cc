// Validation croisee de bout en bout : image -> pipeline -> table. La
// table C++ doit etre identique octet pour octet a celle du prototype
// Python sur les memes images (fixture genere par gen_seg_fixture.py).
//
//     test_table_cross <knn_letters.bin> <knn_digits.bin> <fixtures/seg>

#include <fstream>
#include <iostream>
#include <string>

#include "pyplus/classifier.hh"
#include "pyplus/export_table.hh"
#include "pyplus/model_io.hh"
#include "pyplus/pipeline.hh"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <knn_letters.bin> <knn_digits.bin> <fixtures/seg>\n";
    return 2;
  }
  const std::string seg_dir = argv[3];

  pyplus::KNNClassifier classifier(3, true);
  classifier.fit(pyplus::load_model(argv[1]));
  pyplus::KNNClassifier digit_classifier(3, true);
  digit_classifier.fit(pyplus::load_model(argv[2]));

  std::ifstream fixture(seg_dir + "/expected_tables.txt");
  if (!fixture) {
    std::cerr << "fixture introuvable dans " << seg_dir
              << " (generer avec gen_seg_fixture.py)\n";
    return 2;
  }

  int n_images = 0;
  fixture >> n_images;
  {
    std::string rest;
    std::getline(fixture, rest);
  }

  int n_mismatch = 0;
  for (int i = 0; i < n_images; ++i) {
    std::string name;
    std::getline(fixture, name);
    std::string n_lines_str;
    std::getline(fixture, n_lines_str);
    const int n_lines = std::stoi(n_lines_str);
    std::string expected;
    for (int l = 0; l < n_lines; ++l) {
      std::string line;
      std::getline(fixture, line);
      expected += line + "\n";
    }
    if (!fixture) {
      std::cerr << "fixture tronque a l'image " << i << "\n";
      return 2;
    }

    const pyplus::AutomatonResult result = pyplus::segment_automaton(
        seg_dir + "/" + name, &classifier, &digit_classifier);
    const std::string got = pyplus::result_to_table(result);

    if (got != expected) {
      std::cerr << "=== " << name << " : divergence ===\n"
                << "attendu :\n"
                << expected << "obtenu :\n"
                << got << "\n";
      ++n_mismatch;
    } else {
      std::cout << name << " OK\n";
    }
  }

  if (n_mismatch > 0) {
    std::cerr << n_mismatch << "/" << n_images << " tables divergentes\n";
    return 1;
  }
  std::cout << n_images << " tables identiques a Python\n";
  return 0;
}
