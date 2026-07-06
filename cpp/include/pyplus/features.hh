// Features partagees entre l'entrainement (Python) et l'inference,
// port ligne a ligne de python/src/features.py.
//
// Si ces implementations divergent du prototype, la reconnaissance
// s'effondre (vecteurs incomparables avec le modele entraine) : le test
// features_cross verifie la correspondance sur des glyphes reels.

#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace pyplus {

// Recadre sur la bounding box du trait (pixels > 0).
cv::Mat crop_to_content(const cv::Mat &binary);

// Centre le glyphe dans un carre (preserve ratio) puis redimensionne.
// Entree/sortie : binaire uint8 (trait=255, fond=0).
cv::Mat normalize_crop(const cv::Mat &binary, int target_size = 32);

// HOG maison. img de taille fixe (ex 32x32), uint8.
// Gradients Sobel sur l'image entiere, orientation non signee [0, pi),
// histogramme pondere par la magnitude par cellule, normalisation L2.
std::vector<float> compute_hog(const cv::Mat &img, int cell_size = 8,
                               int n_bins = 9);

} // namespace pyplus
