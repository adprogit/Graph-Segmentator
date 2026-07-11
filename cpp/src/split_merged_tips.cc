#include "pyplus/split_merged_tips.hh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace pyplus
{

namespace
{

Tip make_tip(std::vector<cv::Point> pixels)
{
    Tip tip;
    double sx = 0.0, sy = 0.0;
    int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;
    for (const cv::Point& p : pixels)
    {
        sx += p.x; sy += p.y;
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    const double n = static_cast<double>(pixels.size());
    tip.centroid = cv::Point2d(sx / n, sy / n);
    tip.area = static_cast<int>(pixels.size());
    tip.bbox = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    tip.pixels = std::move(pixels);
    return tip;
}

std::optional<std::pair<Tip, Tip>> try_split_tip(const Tip& tip,
                                                 double median_area)
{
    if (tip.pixels.size() < 8)
        return std::nullopt;

    cv::Mat points(static_cast<int>(tip.pixels.size()), 2, CV_32F);
    for (int i = 0; i < points.rows; ++i)
    {
        points.at<float>(i, 0) = static_cast<float>(tip.pixels[i].x);
        points.at<float>(i, 1) = static_cast<float>(tip.pixels[i].y);
    }

    cv::Mat cluster_ids, centers;
    cv::kmeans(points, 2, cluster_ids,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                20, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<cv::Point> part0, part1;
    for (int i = 0; i < points.rows; ++i)
        (cluster_ids.at<int>(i) == 0 ? part0 : part1)
            .push_back(tip.pixels[static_cast<std::size_t>(i)]);

    const double min_part = 0.35 * median_area;
    if (part0.size() < min_part || part1.size() < min_part)
        return std::nullopt;

    const double dx = centers.at<float>(0, 0) - centers.at<float>(1, 0);
    const double dy = centers.at<float>(0, 1) - centers.at<float>(1, 1);
    const double center_dist = std::sqrt(dx * dx + dy * dy);
    const double typical_size = std::sqrt(median_area);
    if (center_dist < 0.8 * typical_size)
        return std::nullopt;

    return std::make_pair(make_tip(std::move(part0)),
                          make_tip(std::move(part1)));
}

} // namespace

void split_merged_tips(std::vector<Tip>& tips, double merge_area_ratio)
{
    if (tips.size() < 2)
        return;

    std::vector<int> areas;
    areas.reserve(tips.size());
    for (const Tip& t : tips)
        areas.push_back(t.area);
    std::nth_element(areas.begin(), areas.begin() + areas.size() / 2,
                     areas.end());
    const double median_area = areas[areas.size() / 2];

    std::vector<Tip> result;
    result.reserve(tips.size());
    for (Tip& tip : tips)
    {
        if (tip.area <= merge_area_ratio * median_area)
        {
            result.push_back(std::move(tip));
            continue;
        }
        auto split = try_split_tip(tip, median_area);
        if (split)
        {
            result.push_back(std::move(split->first));
            result.push_back(std::move(split->second));
        }
        else
        {
            result.push_back(std::move(tip));
        }
    }
    tips = std::move(result);
}

} // namespace pyplus
