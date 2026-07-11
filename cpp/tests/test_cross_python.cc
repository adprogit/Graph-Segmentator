// Validation croisee : le kNN C++ doit reproduire les predictions du
// prototype Python sur le vrai modele (fixtures generes par
// gen_cross_fixture.py). Lance une fois par modele (lettres, chiffres).
//
//     test_cross_python <knn_letters|digits.bin> <cross_predictions_*.txt>

#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/model_io.hh"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <knn_letters|digits.bin> <cross_predictions_*.txt>\n";
    return 2;
  }

  pyplus::KNNClassifier knn(3, true);
  knn.fit(pyplus::load_model(argv[1]));
  const int dim = knn.model().d;

  std::ifstream fixture(argv[2]);
  if (!fixture) {
    std::cerr << "fixture introuvable : " << argv[2]
              << " (generer avec gen_cross_fixture.py)\n";
    return 2;
  }

  int n_queries = 0;
  int fixture_dim = 0;
  fixture >> n_queries >> fixture_dim;
  if (fixture_dim != dim) {
    std::cerr << "dimension fixture (" << fixture_dim
              << ") != dimension modele (" << dim << ")\n";
    return 2;
  }

  int n_mismatch = 0;
  for (int i = 0; i < n_queries; ++i) {
    std::string expected;
    std::vector<float> query(static_cast<std::size_t>(dim));
    fixture >> expected;
    for (float &v : query)
      fixture >> v;
    if (!fixture) {
      std::cerr << "fixture tronque a la requete " << i << "\n";
      return 2;
    }

    const std::optional<std::string> got = knn.predict(query);
    if (!got || *got != expected) {
      std::cerr << "requete " << i << " : attendu '" << expected
                << "', obtenu '" << got.value_or("<rejet>") << "'\n";
      ++n_mismatch;
    }
  }

  if (n_mismatch > 0) {
    std::cerr << n_mismatch << "/" << n_queries << " divergences\n";
    return 1;
  }
  std::cout << n_queries << " predictions identiques a Python\n";
  return 0;
}
