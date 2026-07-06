// k plus proches voisins, port ligne a ligne de python/src/classifier.py.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "pyplus/model_io.hh"

namespace pyplus
{

class KNNClassifier
{
public:
    // k                nombre de voisins consideres
    // weighted         si true, vote pondere par 1/distance^2
    // reject_threshold si la distance min depasse ce seuil, retourne nullopt
    //                  (crop parasite / hors alphabet)
    explicit KNNClassifier(int k = 3, bool weighted = false,
                           std::optional<float> reject_threshold = std::nullopt);

    // Memorise les donnees d'entrainement (k-NN n'a pas de vrai training).
    void fit(Model model);

    // Predit le label d'un vecteur x de dimension model.d, nullopt si rejet.
    std::optional<std::string> predict_one(const float* x) const;
    std::optional<std::string> predict(const std::vector<float>& x) const;

    const Model& model() const { return model_; }

private:
    std::vector<float> squared_distances(const float* x) const;
    std::vector<int> k_nearest_indices(const std::vector<float>& distances) const;
    std::string vote(const std::vector<int>& neighbor_indices,
                     const std::vector<float>& distances) const;

    int k_;
    bool weighted_;
    std::optional<float> reject_threshold_;
    Model model_;
};

} // namespace pyplus
