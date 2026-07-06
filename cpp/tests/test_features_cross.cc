// Validation croisee : normalize_crop et compute_hog C++ doivent
// reproduire les sorties du prototype Python sur des glyphes reels
// (fixture genere par gen_features_fixture.py).
//
//     test_features_cross <features_expected.txt>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pyplus/features.hh"

namespace {

// Lit "h w" puis h lignes de w caracteres '0'/'1' (1 = 255).
cv::Mat read_binary_image(std::ifstream &f) {
  int h = 0;
  int w = 0;
  f >> h >> w;
  cv::Mat img(h, w, CV_8U);
  for (int y = 0; y < h; ++y) {
    std::string row;
    f >> row;
    for (int x = 0; x < w; ++x)
      img.at<std::uint8_t>(y, x) =
          row[static_cast<std::size_t>(x)] == '1' ? 255 : 0;
  }
  return img;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <features_expected.txt>\n";
    return 2;
  }
  std::ifstream fixture(argv[1]);
  if (!fixture) {
    std::cerr << "fixture introuvable : " << argv[1]
              << " (generer avec gen_features_fixture.py)\n";
    return 2;
  }

  int n_crops = 0;
  fixture >> n_crops;
  int n_mismatch = 0;
  for (int i = 0; i < n_crops; ++i) {
    const cv::Mat raw = read_binary_image(fixture);
    const cv::Mat expected_norm = read_binary_image(fixture);
    std::vector<float> expected_hog(144);
    for (float &v : expected_hog)
      fixture >> v;
    if (!fixture) {
      std::cerr << "fixture tronque au crop " << i << "\n";
      return 2;
    }

    const cv::Mat norm = pyplus::normalize_crop(raw);
    if (cv::countNonZero(norm != expected_norm) != 0) {
      std::cerr << "crop " << i << " : normalize_crop divergent\n";
      ++n_mismatch;
      continue;
    }

    const std::vector<float> hog = pyplus::compute_hog(norm);
    float max_err = 0.0f;
    for (std::size_t j = 0; j < hog.size(); ++j)
      max_err = std::max(max_err, std::abs(hog[j] - expected_hog[j]));
    if (hog.size() != expected_hog.size() || max_err > 1e-5f) {
      std::cerr << "crop " << i << " : HOG divergent (err max " << max_err
                << ")\n";
      ++n_mismatch;
    }
  }

  if (n_mismatch > 0) {
    std::cerr << n_mismatch << "/" << n_crops << " divergences\n";
    return 1;
  }
  std::cout << n_crops << " crops identiques a Python\n";
  return 0;
}
