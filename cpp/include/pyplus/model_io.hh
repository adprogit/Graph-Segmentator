// Chargement du modele kNN depuis le format binaire portable
// (cf. python/src/model_io.py). Layout little-endian :
//     [int32] N   nombre d'exemples
//     [int32] D   dimension des vecteurs de features
//     [int32] C   nombre de classes
//     [N * D * float32] X   vecteurs, ligne par ligne
//     [N * int32]       y   index de classe de chaque exemple (0..C-1)
//     [C blocs]  labels : [int32 len][len octets utf-8]

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pyplus {

struct Model {
  int n = 0;                        // nombre d'exemples
  int d = 0;                        // dimension des features
  std::vector<float> x;             // (n * d), ligne par ligne
  std::vector<int> y;               // (n,) index de classe
  std::vector<std::string> classes; // labels utf-8, tries

  const float *row(int i) const {
    return x.data() + static_cast<std::size_t>(i) * d;
  }

  const std::string &label(int i) const {
    return classes[static_cast<std::size_t>(y[static_cast<std::size_t>(i)])];
  }
};

// Lance std::runtime_error si le fichier est absent ou tronque.
Model load_model(const std::string &path);

void save_model(const std::string &path, const Model &model);

} // namespace pyplus
