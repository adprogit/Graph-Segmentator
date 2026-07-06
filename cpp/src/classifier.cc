#include "pyplus/classifier.hh"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace pyplus {

KNNClassifier::KNNClassifier(int k, bool weighted,
                             std::optional<float> reject_threshold)
    : k_(k), weighted_(weighted), reject_threshold_(reject_threshold) {}

void KNNClassifier::fit(Model model) { model_ = std::move(model); }

// Distances au carre entre x (d,) et tous les points (sqrt inutile).
std::vector<float> KNNClassifier::squared_distances(const float *x) const {
  std::vector<float> distances(static_cast<std::size_t>(model_.n));
  for (int i = 0; i < model_.n; ++i) {
    const float *row = model_.row(i);
    double acc = 0.0;
    for (int j = 0; j < model_.d; ++j) {
      const double diff = static_cast<double>(row[j]) - x[j];
      acc += diff * diff;
    }
    distances[static_cast<std::size_t>(i)] = static_cast<float>(acc);
  }
  return distances;
}

// Indices des k plus proches points (pas forcement tries).
std::vector<int>
KNNClassifier::k_nearest_indices(const std::vector<float> &distances) const {
  const int k = std::min<int>(k_, static_cast<int>(distances.size()));
  std::vector<int> indices(distances.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::nth_element(indices.begin(), indices.begin() + (k - 1), indices.end(),
                   [&](int a, int b) {
                     return distances[static_cast<std::size_t>(a)] <
                            distances[static_cast<std::size_t>(b)];
                   });
  indices.resize(static_cast<std::size_t>(k));
  return indices;
}

// Determine le label a partir des voisins.
// - vote simple : label majoritaire (egalite -> premier par ordre trie,
//   comme np.unique + argmax)
// - vote pondere : chaque voisin pese 1/distance^2 (egalite -> premier
//   voisin rencontre, comme max() sur un dict Python)
std::string KNNClassifier::vote(const std::vector<int> &neighbor_indices,
                                const std::vector<float> &distances) const {
  if (weighted_) {
    std::vector<std::pair<std::string, double>> weights;
    for (int idx : neighbor_indices) {
      const std::string &label = model_.label(idx);
      auto it = std::find_if(weights.begin(), weights.end(),
                             [&](const auto &w) { return w.first == label; });
      if (it == weights.end())
        it = weights.insert(weights.end(), {label, 0.0});
      it->second +=
          1.0 / (static_cast<double>(distances[static_cast<std::size_t>(idx)]) +
                 1e-8);
    }
    const auto best = std::max_element(
        weights.begin(), weights.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    return best->first;
  }

  std::map<std::string, int> counts; // trie, comme np.unique
  for (int idx : neighbor_indices)
    ++counts[model_.label(idx)];
  const auto best = std::max_element(
      counts.begin(), counts.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });
  return best->first;
}

std::optional<std::string> KNNClassifier::predict_one(const float *x) const {
  if (model_.n == 0)
    throw std::runtime_error("classifieur non entraine (fit manquant)");

  const std::vector<float> distances = squared_distances(x);
  if (reject_threshold_) {
    const float min_dist =
        *std::min_element(distances.begin(), distances.end());
    if (min_dist > *reject_threshold_ * *reject_threshold_)
      return std::nullopt;
  }
  return vote(k_nearest_indices(distances), distances);
}

std::optional<std::string>
KNNClassifier::predict(const std::vector<float> &x) const {
  if (static_cast<int>(x.size()) != model_.d)
    throw std::runtime_error("dimension du vecteur != dimension du modele");
  return predict_one(x.data());
}

} // namespace pyplus
