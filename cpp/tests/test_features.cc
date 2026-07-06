// Tests unitaires des features (normalize_crop + HOG) sur des images
// construites a la main.

#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "pyplus/features.hh"

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " echec : " #cond "\n";      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  // crop_to_content : bounding box du trait
  cv::Mat img = cv::Mat::zeros(20, 30, CV_8U);
  img(cv::Rect(5, 8, 10, 4)) = 255; // bloc 10x4 en (5,8)
  const cv::Mat content = pyplus::crop_to_content(img);
  CHECK(content.rows == 4 && content.cols == 10);
  CHECK(cv::countNonZero(content) == 40);

  // image vide : retournee telle quelle, puis normalisee en zeros
  const cv::Mat blank = cv::Mat::zeros(20, 30, CV_8U);
  CHECK(pyplus::crop_to_content(blank).size() == blank.size());
  const cv::Mat norm_blank = pyplus::normalize_crop(blank);
  CHECK(norm_blank.rows == 32 && norm_blank.cols == 32);
  CHECK(cv::countNonZero(norm_blank) == 0);

  // normalize_crop : contenu centre, ratio preserve, sortie 32x32
  const cv::Mat norm = pyplus::normalize_crop(img);
  CHECK(norm.rows == 32 && norm.cols == 32);
  // bloc 10x4 -> carre 10x10 avec bandes vides en haut/bas -> apres
  // resize, les lignes du haut et du bas restent vides
  CHECK(cv::countNonZero(norm.row(0)) == 0);
  CHECK(cv::countNonZero(norm.row(31)) == 0);
  CHECK(cv::countNonZero(norm.row(16)) > 0);

  // HOG : dimension (32/8)^2 cellules * 9 bins = 144, norme L2 = 1
  const std::vector<float> hog = pyplus::compute_hog(norm);
  CHECK(hog.size() == 144);
  double norm_sq = 0.0;
  for (float v : hog)
    norm_sq += static_cast<double>(v) * v;
  CHECK(std::abs(std::sqrt(norm_sq) - 1.0) < 1e-5);

  // image sans trait -> vecteur nul (pas de division par 0)
  const std::vector<float> hog_blank = pyplus::compute_hog(norm_blank);
  CHECK(hog_blank.size() == 144);
  for (float v : hog_blank)
    CHECK(v == 0.0f);

  // bord vertical : gradient horizontal -> orientation 0 -> tout le
  // poids dans le bin 0 de chaque cellule traversee par le bord
  cv::Mat edge = cv::Mat::zeros(32, 32, CV_8U);
  edge(cv::Rect(16, 0, 16, 32)) = 255;
  const std::vector<float> hog_edge = pyplus::compute_hog(edge);
  for (std::size_t c = 0; c < 16; ++c)
    for (std::size_t b = 1; b < 9; ++b)
      CHECK(hog_edge[c * 9 + b] == 0.0f);
  CHECK(hog_edge[9] > 0.0f || hog_edge[0] > 0.0f);

  std::cout << "features OK\n";
  return 0;
}
