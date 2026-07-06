#include "pyplus/features.hh"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace pyplus
{

cv::Mat crop_to_content(const cv::Mat& binary)
{
    std::vector<cv::Point> nonzero;
    cv::findNonZero(binary, nonzero);
    if (nonzero.empty())
        return binary;
    const cv::Rect bbox = cv::boundingRect(nonzero);
    return binary(bbox);
}

cv::Mat normalize_crop(const cv::Mat& binary, int target_size)
{
    const cv::Mat content = crop_to_content(binary);
    const int h = content.rows;
    const int w = content.cols;
    if (h == 0 || w == 0)
        return cv::Mat::zeros(target_size, target_size, CV_8U);

    const int side = std::max(h, w);
    cv::Mat square = cv::Mat::zeros(side, side, CV_8U);
    const int yo = (side - h) / 2;
    const int xo = (side - w) / 2;
    content.copyTo(square(cv::Rect(xo, yo, w, h)));

    cv::Mat resized;
    cv::resize(square, resized, cv::Size(target_size, target_size), 0, 0,
               cv::INTER_NEAREST);
    return resized;
}

std::vector<float> compute_hog(const cv::Mat& img, int cell_size, int n_bins)
{
    if (img.empty() || img.type() != CV_8U)
        throw std::runtime_error("compute_hog attend une image uint8");

    cv::Mat img_f;
    img.convertTo(img_f, CV_32F);
    cv::Mat gx, gy;
    cv::Sobel(img_f, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img_f, gy, CV_32F, 0, 1, 3);

    const int h = img.rows;
    const int w = img.cols;
    const float pi = static_cast<float>(CV_PI);
    const float bin_width = pi / static_cast<float>(n_bins);

    std::vector<float> features;
    for (int i = 0; i + cell_size <= h; i += cell_size)
    {
        for (int j = 0; j + cell_size <= w; j += cell_size)
        {
            std::vector<float> hist(static_cast<std::size_t>(n_bins), 0.0f);
            for (int y = i; y < i + cell_size; ++y)
            {
                const float* gx_row = gx.ptr<float>(y);
                const float* gy_row = gy.ptr<float>(y);
                for (int x = j; x < j + cell_size; ++x)
                {
                    const float dx = gx_row[x];
                    const float dy = gy_row[x];
                    const float magnitude = std::sqrt(dx * dx + dy * dy);
                    // equivalent de numpy `arctan2(gy, gx) % pi` : [0, pi)
                    // gradient horizontal -> angle 0 exact : atan2(0, x<0)
                    // vaut pi-1ulp sur macOS mais pi pile sur Linux, ce qui
                    // fait basculer le bin (8 vs 0)
                    float orientation = 0.0f;
                    if (dy != 0.0f)
                    {
                        orientation = std::fmod(std::atan2(dy, dx), pi);
                        if (orientation < 0.0f)
                            orientation += pi;
                    }
                    int b = static_cast<int>(orientation / bin_width);
                    if (b >= n_bins)
                        b = n_bins - 1;
                    hist[static_cast<std::size_t>(b)] += magnitude;
                }
            }
            features.insert(features.end(), hist.begin(), hist.end());
        }
    }

    double norm_sq = 0.0;
    for (float v : features)
        norm_sq += static_cast<double>(v) * v;
    const float norm = static_cast<float>(std::sqrt(norm_sq));
    if (norm > 0.0f)
        for (float& v : features)
            v /= norm;
    return features;
}

} // namespace pyplus
