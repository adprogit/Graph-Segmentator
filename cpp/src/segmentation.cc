#include "pyplus/segmentation.hh"
#include "pyplus/split_merged_tips.hh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <utility>

namespace pyplus {

namespace {

constexpr std::pair<int, int> kNeighbors[] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1},
};

// Pixels d'une composante, ordre ligne a ligne (equivalent np.where).
std::vector<cv::Point> component_pixels(const cv::Mat &labels_image,
                                        int component_index,
                                        const cv::Rect &bbox) {
  std::vector<cv::Point> pixels;
  for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
    const std::int32_t *row = labels_image.ptr<std::int32_t>(y);
    for (int x = bbox.x; x < bbox.x + bbox.width; ++x)
      if (row[x] == component_index)
        pixels.emplace_back(x, y);
  }
  return pixels;
}

} // namespace

cv::Mat load_image(const std::string &path) {
  return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

cv::Mat to_binary(const cv::Mat &gray) {
  cv::Mat binary;
  cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
  return binary;
}

std::vector<State> segment_states(const cv::Mat &binary_image,
                                  double relative_min_area,
                                  double min_area_cap) {
  const int image_height = binary_image.rows;
  const int image_width = binary_image.cols;
  const double min_area =
      std::min(image_width * image_height * relative_min_area, min_area_cap);

  cv::Mat background_image;
  cv::bitwise_not(binary_image, background_image);

  cv::Mat labels_image;
  cv::Mat stats;
  cv::Mat centroids;
  const int n_components = cv::connectedComponentsWithStats(
      background_image, labels_image, stats, centroids, 8);

  std::vector<State> state_candidates;
  for (int i = 1; i < n_components; ++i) {
    const int bx = stats.at<std::int32_t>(i, cv::CC_STAT_LEFT);
    const int by = stats.at<std::int32_t>(i, cv::CC_STAT_TOP);
    const int bw = stats.at<std::int32_t>(i, cv::CC_STAT_WIDTH);
    const int bh = stats.at<std::int32_t>(i, cv::CC_STAT_HEIGHT);
    const int component_area = stats.at<std::int32_t>(i, cv::CC_STAT_AREA);

    const bool touches_border =
        bx == 0 || by == 0 || bx + bw == image_width || by + bh == image_height;
    if (touches_border)
      continue;

    if (component_area < min_area)
      continue;

    cv::Mat component_mask;
    cv::compare(labels_image, i, component_mask, cv::CMP_EQ);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(component_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);
    if (contours.empty())
      continue;

    const double perimeter = cv::arcLength(contours[0], true);
    if (perimeter == 0.0)
      continue;

    const double circularity =
        4.0 * CV_PI * component_area / (perimeter * perimeter);
    if (circularity < 0.85)
      continue;

    const double approximate_radius = (bw + bh) / 4.0;
    const double centroid_x = centroids.at<double>(i, 0);
    const double centroid_y = centroids.at<double>(i, 1);

    cv::Mat name_crop = binary_image(cv::Rect(bx, by, bw, bh)).clone();
    const cv::Point local_center(static_cast<int>(centroid_x) - bx,
                                 static_cast<int>(centroid_y) - by);
    cv::Mat disk = cv::Mat::zeros(name_crop.size(), name_crop.type());
    cv::circle(disk, local_center, static_cast<int>(approximate_radius) - 2,
               255, -1);
    cv::bitwise_and(name_crop, disk, name_crop);

    State state;
    state.center_x = centroid_x;
    state.center_y = centroid_y;
    state.radius = approximate_radius;
    state.outer_radius = approximate_radius;
    state.accepting = false;
    state.area = component_area;
    state.name_crop = name_crop;
    state_candidates.push_back(std::move(state));
  }

  return state_candidates;
}

cv::Mat remove_states_from_img(const cv::Mat &binary_image,
                               const std::vector<State> &states, int padding) {
  cv::Mat cleaned_image = binary_image.clone();
  for (const State &state : states) {
    const cv::Point center(static_cast<int>(state.center_x),
                           static_cast<int>(state.center_y));
    const int erase_radius = static_cast<int>(state.outer_radius) + padding;
    cv::circle(cleaned_image, center, erase_radius, 0, -1);
  }
  return cleaned_image;
}

void merge_states_and_acceptants(std::vector<State> &states_first_pass,
                                 const std::vector<State> &residual_circles,
                                 double center_tolerance) {
  for (State &state : states_first_pass) {
    state.accepting = false;
    for (const State &residual : residual_circles) {
      const double dx = state.center_x - residual.center_x;
      const double dy = state.center_y - residual.center_y;
      const double distance = std::sqrt(dx * dx + dy * dy);
      if (distance <= center_tolerance) {
        state.accepting = true;
        state.outer_radius = residual.radius;
        break;
      }
    }
  }
}

std::vector<Tip> find_triangles(const cv::Mat &img_cleaned) {
  const int image_height = img_cleaned.rows;
  const int image_width = img_cleaned.cols;

  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
  cv::Mat opened;
  cv::morphologyEx(img_cleaned, opened, cv::MORPH_OPEN, kernel);

  cv::Mat labels_image;
  cv::Mat stats;
  cv::Mat centroids;
  const int n_components = cv::connectedComponentsWithStats(
      opened, labels_image, stats, centroids, 8);

  std::vector<Tip> tips;
  for (int i = 1; i < n_components; ++i) {
    const int bx = stats.at<std::int32_t>(i, cv::CC_STAT_LEFT);
    const int by = stats.at<std::int32_t>(i, cv::CC_STAT_TOP);
    const int bw = stats.at<std::int32_t>(i, cv::CC_STAT_WIDTH);
    const int bh = stats.at<std::int32_t>(i, cv::CC_STAT_HEIGHT);
    const int component_area = stats.at<std::int32_t>(i, cv::CC_STAT_AREA);

    const bool touches_border =
        bx == 0 || by == 0 || bx + bw == image_width || by + bh == image_height;

    if (touches_border)
      continue;

    if (component_area < 6)
      continue;

    cv::Mat component_mask;
    cv::compare(labels_image, i, component_mask, cv::CMP_EQ);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(component_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);

    if (contours.empty())
      continue;
    const double perimeter = cv::arcLength(contours[0], true);

    if (perimeter == 0.0)
      continue;

    const double circularity =
        4.0 * CV_PI * component_area / (perimeter * perimeter);
    if (circularity > 0.98)
      continue;
    Tip tip;
    tip.centroid =
        cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
    tip.area = component_area;
    tip.bbox = cv::Rect(bx, by, bw, bh);
    tip.pixels = component_pixels(labels_image, i, tip.bbox);
    tips.push_back(std::move(tip));
  }
  split_merged_tips(tips);
  return tips;
}

namespace {

struct TipAnalysis {
  cv::Point2d apex;
  cv::Point2d direction;
  std::optional<int> dest;
};

// Apex (point du blob le plus proche d'un etat), direction (centroide ->
// centre de l'etat destination) et index de destination.
TipAnalysis analyze_tip(const std::vector<cv::Point> &tip_pixels,
                        const cv::Point2d &centroid,
                        const std::vector<State> &states) {
  TipAnalysis result;
  double best_distance = std::numeric_limits<double>::infinity();
  std::optional<cv::Point2d> best_state_center;

  for (std::size_t state_index = 0; state_index < states.size();
       ++state_index) {
    const State &state = states[state_index];
    // argmin sur les pixels du blob (premier minimum, ordre ligne a ligne)
    double min_distance = std::numeric_limits<double>::infinity();
    cv::Point2d min_pixel;
    for (const cv::Point &p : tip_pixels) {
      const double dx = p.x - state.center_x;
      const double dy = p.y - state.center_y;
      const double distance = std::sqrt(dx * dx + dy * dy);
      if (distance < min_distance) {
        min_distance = distance;
        min_pixel = cv::Point2d(p.x, p.y);
      }
    }
    if (min_distance < best_distance) {
      best_distance = min_distance;
      result.apex = min_pixel;
      result.dest = static_cast<int>(state_index);
      best_state_center = cv::Point2d(state.center_x, state.center_y);
    }
  }

  double direction_x;
  double direction_y;
  if (best_state_center) {
    direction_x = best_state_center->x - centroid.x;
    direction_y = best_state_center->y - centroid.y;
  } else {
    direction_x = result.apex.x - centroid.x;
    direction_y = result.apex.y - centroid.y;
  }
  const double norm =
      std::sqrt(direction_x * direction_x + direction_y * direction_y);
  result.direction = norm > 0.0
                         ? cv::Point2d(direction_x / norm, direction_y / norm)
                         : cv::Point2d(0.0, 0.0);
  return result;
}

// Index de l'etat atteint par (x, y), ou nullopt.
std::optional<int> reached_state(int x, int y, const std::vector<State> &states,
                                 int step, std::optional<int> origin_dest_index,
                                 int min_steps_before_return) {
  for (std::size_t idx = 0; idx < states.size(); ++idx) {
    if (origin_dest_index && static_cast<int>(idx) == *origin_dest_index &&
        step < min_steps_before_return)
      continue;
    const State &s = states[idx];
    const double r = s.outer_radius + 8.0;
    const double dx = x - s.center_x;
    const double dy = y - s.center_y;
    if (dx * dx + dy * dy <= r * r)
      return static_cast<int>(idx);
  }
  return std::nullopt;
}

// Voisin blanc non visite le mieux aligne avec direction, ou nullopt.
std::optional<cv::Point>
best_neighbor(int cx, int cy, const cv::Mat &img,
              const std::set<std::pair<int, int>> &visited,
              const cv::Point2d &direction) {
  const int height = img.rows;
  const int width = img.cols;

  std::optional<cv::Point> best;
  double best_score = -std::numeric_limits<double>::infinity();
  for (const auto &[dx, dy] : kNeighbors) {
    const int nx = cx + dx;
    const int ny = cy + dy;
    if (!(0 <= ny && ny < height && 0 <= nx && nx < width))
      continue;
    if (img.at<std::uint8_t>(ny, nx) == 0)
      continue;
    if (visited.count({nx, ny}))
      continue;

    const double vx = nx - cx;
    const double vy = ny - cy;
    const double n = std::sqrt(vx * vx + vy * vy);
    const double score =
        n != 0.0 ? (vx * direction.x + vy * direction.y) / n : -1.0;
    if (score > best_score) // egalite -> premier voisin (comme max())
    {
      best_score = score;
      best = cv::Point(nx, ny);
    }
  }
  return best;
}

// Suit le trace depuis start_pt ; retourne (chemin, etat atteint | nullopt).
std::pair<std::vector<cv::Point>, std::optional<int>>
follow_line(const cv::Mat &img, const cv::Point2d &start_pt,
            cv::Point2d direction, const std::vector<State> &states,
            std::optional<int> origin_dest_index = std::nullopt,
            int max_steps = 2000, int smoothing_window = 5,
            int min_steps_before_return = 8) {
  const int height = img.rows;
  const int width = img.cols;

  int sx = static_cast<int>(start_pt.x);
  int sy = static_cast<int>(start_pt.y);
  if (!(0 <= sy && sy < height && 0 <= sx && sx < width) ||
      img.at<std::uint8_t>(sy, sx) == 0) {
    // point de depart hors trait : pixel blanc le plus proche
    double best_d = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int y = 0; y < height; ++y) {
      const std::uint8_t *row = img.ptr<std::uint8_t>(y);
      for (int x = 0; x < width; ++x) {
        if (row[x] == 0)
          continue;
        const double d = (x - start_pt.x) * (x - start_pt.x) +
                         (y - start_pt.y) * (y - start_pt.y);
        if (d < best_d) {
          best_d = d;
          sx = x;
          sy = y;
          found = true;
        }
      }
    }
    if (!found)
      return {{}, std::nullopt};
  }

  std::vector<cv::Point> chemin = {cv::Point(sx, sy)};
  std::set<std::pair<int, int>> visited = {{sx, sy}};

  for (int step = 0; step < max_steps; ++step) {
    const cv::Point current = chemin.back();

    const std::optional<int> hit =
        reached_state(current.x, current.y, states, step, origin_dest_index,
                      min_steps_before_return);
    if (hit)
      return {chemin, hit};

    const std::optional<cv::Point> nxt =
        best_neighbor(current.x, current.y, img, visited, direction);
    if (!nxt)
      return {chemin, std::nullopt}; // cul-de-sac

    visited.insert({nxt->x, nxt->y});
    chemin.push_back(*nxt);

    const std::size_t ref_index =
        chemin.size() > static_cast<std::size_t>(smoothing_window)
            ? chemin.size() - static_cast<std::size_t>(smoothing_window)
            : 0;
    const cv::Point ref = chemin[ref_index];
    const double dx = nxt->x - ref.x;
    const double dy = nxt->y - ref.y;
    const double n = std::sqrt(dx * dx + dy * dy);
    if (n != 0.0)
      direction = cv::Point2d(dx / n, dy / n);
  }

  return {chemin, std::nullopt};
}

// Etat le plus proche dans le cone defini par direction (fallback).
std::optional<int> find_state_in_direction(
    const cv::Point2d &origin, const cv::Point2d &direction,
    const std::vector<State> &states, std::optional<int> exclude = std::nullopt,
    double cos_min = 0.5, std::optional<double> max_distance = std::nullopt) {
  std::optional<int> best;
  double best_distance_squared = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < states.size(); ++index) {
    if (exclude && static_cast<int>(index) == *exclude)
      continue;
    const double delta_x = states[index].center_x - origin.x;
    const double delta_y = states[index].center_y - origin.y;
    const double norm_delta = std::sqrt(delta_x * delta_x + delta_y * delta_y);
    if (norm_delta == 0.0)
      continue;
    const double cos_theta =
        (delta_x * direction.x + delta_y * direction.y) / norm_delta;
    if (cos_theta < cos_min)
      continue;
    const double distance_squared = delta_x * delta_x + delta_y * delta_y;
    if (max_distance && distance_squared > *max_distance * *max_distance)
      continue;
    if (distance_squared < best_distance_squared) {
      best_distance_squared = distance_squared;
      best = static_cast<int>(index);
    }
  }
  return best;
}

} // namespace
struct BranchResult {
  std::vector<cv::Point> chemin;
  int source;
};

std::vector<cv::Point>
branch_candidates(int cx, int cy, const cv::Mat &img,
                  const std::set<std::pair<int, int>> &visited,
                  const cv::Point2d &direction, double min_align,
                  double gap_deg) {
  const int height = img.rows;
  const int width = img.cols;

  struct Cand {
    double angle;
    cv::Point pt;
    double align;
  };
  std::vector<Cand> cands;
  for (const auto &[dx, dy] : kNeighbors) {
    const int nx = cx + dx, ny = cy + dy;
    if (!(0 <= ny && ny < height && 0 <= nx && nx < width))
      continue;
    if (img.at<std::uint8_t>(ny, nx) == 0)
      continue;
    if (visited.count({nx, ny}))
      continue;
    const double vx = nx - cx, vy = ny - cy;
    const double n = std::sqrt(vx * vx + vy * vy);
    if (n == 0.0)
      continue;
    const double align = (vx * direction.x + vy * direction.y) / n;
    if (align < min_align)
      continue; // ecarte perpendiculaire / arriere
    cands.push_back({std::atan2(vy, vx), cv::Point(nx, ny), align});
  }
  if (cands.empty())
    return {};

  std::sort(cands.begin(), cands.end(),
            [](const Cand &a, const Cand &b) { return a.angle < b.angle; });
  const double gap = gap_deg * CV_PI / 180.0;

  std::vector<cv::Point> reps;
  double cluster_best_align = -2.0;
  cv::Point cluster_best_pt;
  double prev_angle = cands.front().angle;
  bool open = false;
  auto flush = [&]() {
    if (open)
      reps.push_back(cluster_best_pt);
  };

  for (const Cand &c : cands) {
    if (open && c.angle - prev_angle > gap) {
      flush();
      cluster_best_align = -2.0;
    }
    if (c.align > cluster_best_align) {
      cluster_best_align = c.align;
      cluster_best_pt = c.pt;
    }
    prev_angle = c.angle;
    open = true;
  }
  flush();
  return reps;
}

std::vector<BranchResult>
follow_line_branches(const cv::Mat &img, const cv::Point2d &start_pt,
                     cv::Point2d initial_dir, const std::vector<State> &states,
                     std::optional<int> origin_dest_index, int max_steps = 2000,
                     int smoothing_window = 5, int min_steps_before_return = 8,
                     double fork_min_align = 0.15, double fork_gap_deg = 20.0,
                     int max_branches = 4) {
  const int height = img.rows, width = img.cols;

  int sx = static_cast<int>(start_pt.x), sy = static_cast<int>(start_pt.y);
  if (!(0 <= sy && sy < height && 0 <= sx && sx < width) ||
      img.at<std::uint8_t>(sy, sx) == 0) {
    double best_d = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int y = 0; y < height; ++y) {
      const std::uint8_t *row = img.ptr<std::uint8_t>(y);
      for (int x = 0; x < width; ++x) {
        if (row[x] == 0)
          continue;
        const double d = (x - start_pt.x) * (x - start_pt.x) +
                         (y - start_pt.y) * (y - start_pt.y);
        if (d < best_d) {
          best_d = d;
          sx = x;
          sy = y;
          found = true;
        }
      }
    }
    if (!found)
      return {};
  }

  struct Frame {
    cv::Point pos;
    cv::Point2d dir;
    std::vector<cv::Point> chemin;
  };
  std::vector<Frame> stack;
  stack.push_back({cv::Point(sx, sy), initial_dir, {cv::Point(sx, sy)}});
  std::set<std::pair<int, int>> visited = {{sx, sy}};

  std::vector<BranchResult> results;
  std::set<int> seen_sources;
  int branches_used = 0;

  while (!stack.empty()) {
    Frame f = std::move(stack.back());
    stack.pop_back();
    cv::Point2d direction = f.dir;

    for (int step = 0; step < max_steps; ++step) {
      const cv::Point current = f.chemin.back();

      const std::optional<int> hit = reached_state(
          current.x, current.y, states, static_cast<int>(f.chemin.size()),
          origin_dest_index, min_steps_before_return);
      if (hit) {
        if (!seen_sources.count(*hit)) {
          seen_sources.insert(*hit);
          results.push_back({f.chemin, *hit});
        }
        break;
      }

      const std::vector<cv::Point> reps =
          branch_candidates(current.x, current.y, img, visited, direction,
                            fork_min_align, fork_gap_deg);

      if (reps.empty())
        break; // cul-de-sac

      if (reps.size() == 1 || branches_used >= max_branches) {
        const cv::Point nxt = reps.front();
        visited.insert({nxt.x, nxt.y});
        f.chemin.push_back(nxt);
        const std::size_t ri =
            f.chemin.size() > static_cast<std::size_t>(smoothing_window)
                ? f.chemin.size() - smoothing_window
                : 0;
        const cv::Point ref = f.chemin[ri];
        const double dx = nxt.x - ref.x, dy = nxt.y - ref.y;
        const double n = std::sqrt(dx * dx + dy * dy);
        if (n != 0.0)
          direction = cv::Point2d(dx / n, dy / n);
      } else {
        for (const cv::Point &rep : reps) {
          if (visited.count({rep.x, rep.y}))
            continue;
          visited.insert({rep.x, rep.y});
          std::vector<cv::Point> ch = f.chemin;
          ch.push_back(rep);
          const double dx = rep.x - current.x, dy = rep.y - current.y;
          const double n = std::sqrt(dx * dx + dy * dy);
          const cv::Point2d d =
              n != 0.0 ? cv::Point2d(dx / n, dy / n) : direction;
          stack.push_back({rep, d, std::move(ch)});
          ++branches_used;
        }
        break;
      }
    }
  }
  return results;
}

AdjacencyResult build_adjacency_matrix(const std::vector<State> &states,
                                       const std::vector<Tip> &triangles,
                                       const cv::Mat &img_clean_final) {
  AdjacencyResult result;
  result.matrix.assign(states.size(), std::vector<int>(states.size(), -1));

  for (const Tip &triangle : triangles) {
    if (triangle.pixels.empty())
      continue;
    const TipAnalysis tip =
        analyze_tip(triangle.pixels, triangle.centroid, states);
    if (!tip.dest)
      continue;

    const double base_x = 2.0 * triangle.centroid.x - tip.apex.x;
    const double base_y = 2.0 * triangle.centroid.y - tip.apex.y;
    const double init_dx = base_x - tip.apex.x;
    const double init_dy = base_y - tip.apex.y;
    const double init_norm = std::sqrt(init_dx * init_dx + init_dy * init_dy);
    const cv::Point2d init_dir =
        init_norm > 0.0 ? cv::Point2d(init_dx / init_norm, init_dy / init_norm)
                        : cv::Point2d(-tip.direction.x, -tip.direction.y);

    cv::Mat img_suivi = img_clean_final.clone();
    for (const cv::Point &p : triangle.pixels)
      img_suivi.at<std::uint8_t>(p.y, p.x) = 0;
    std::vector<BranchResult> branches = follow_line_branches(
        img_suivi, cv::Point2d(base_x, base_y), init_dir, states, tip.dest);
    if (branches.empty()) {
      // repli geometrique (une seule source presumee)
      const cv::Point2d opposite(-tip.direction.x, -tip.direction.y);
      const std::optional<int> src = find_state_in_direction(
          triangle.centroid, opposite, states, tip.dest);
      if (!src) {
        if (!result.initial)
          result.initial = tip.dest; // fleche initiale
        continue;
      }
      branches.push_back({{}, *src});
    }

    // une arete par branche (source distincte), meme destination
    for (BranchResult &br : branches) {
      std::cerr << "branch found\n";
      Arrow edge;
      edge.source = br.source;
      edge.dest = *tip.dest;
      edge.chemin = std::move(br.chemin);
      result.arrows.push_back(std::move(edge));
      result.matrix[static_cast<std::size_t>(br.source)]
                   [static_cast<std::size_t>(*tip.dest)] =
          static_cast<int>(result.arrows.size()) - 1;
    }
  }

  return result;
}

cv::Mat isolate_labels(const cv::Mat &img_clean_final,
                       const std::vector<Tip> &tips,
                       const std::vector<Arrow> &arrows) {
  cv::Mat labels_image;
  cv::Mat stats;
  cv::Mat centroids;
  cv::connectedComponentsWithStats(img_clean_final, labels_image, stats,
                                   centroids, 8);

  std::set<std::int32_t> arrow_component_ids;
  for (const Arrow &arrow : arrows) {
    for (const cv::Point &p : arrow.chemin) {
      if (0 <= p.y && p.y < labels_image.rows && 0 <= p.x &&
          p.x < labels_image.cols) {
        const std::int32_t label_id = labels_image.at<std::int32_t>(p.y, p.x);
        if (label_id != 0)
          arrow_component_ids.insert(label_id);
      }
    }
  }
  for (const Tip &tip : tips) {
    if (!tip.pixels.empty()) {
      const cv::Point &p = tip.pixels.front();
      const std::int32_t label_id = labels_image.at<std::int32_t>(p.y, p.x);
      if (label_id != 0)
        arrow_component_ids.insert(label_id);
    }
  }

  cv::Mat img_labels = img_clean_final.clone();
  for (int y = 0; y < img_labels.rows; ++y) {
    const std::int32_t *label_row = labels_image.ptr<std::int32_t>(y);
    std::uint8_t *out_row = img_labels.ptr<std::uint8_t>(y);
    for (int x = 0; x < img_labels.cols; ++x)
      if (arrow_component_ids.count(label_row[x]))
        out_row[x] = 0;
  }
  return img_labels;
}

cv::Mat extract_label_crop(const cv::Mat &img_labels, const cv::Rect &bbox,
                           int padding, int target_size) {
  const int h = img_labels.rows;
  const int w = img_labels.cols;
  const int y0 = std::max(0, bbox.y - padding);
  const int y1 = std::min(h, bbox.y + bbox.height + padding);
  const int x0 = std::max(0, bbox.x - padding);
  const int x1 = std::min(w, bbox.x + bbox.width + padding);
  const cv::Mat raw = img_labels(cv::Rect(x0, y0, x1 - x0, y1 - y0));

  const int ch = raw.rows;
  const int cw = raw.cols;
  if (ch == 0 || cw == 0)
    return cv::Mat::zeros(target_size, target_size, CV_8U);
  const int side = std::max(ch, cw);
  cv::Mat square = cv::Mat::zeros(side, side, CV_8U);
  const int yo = (side - ch) / 2;
  const int xo = (side - cw) / 2;
  raw.copyTo(square(cv::Rect(xo, yo, cw, ch)));

  cv::Mat resized;
  cv::resize(square, resized, cv::Size(target_size, target_size), 0, 0,
             cv::INTER_NEAREST);
  return resized;
}

void assign_labels_to_arrows(const cv::Mat &img_labels,
                             std::vector<Arrow> &arrows, int min_area) {
  for (Arrow &arrow : arrows)
    arrow.labels.clear();

  cv::Mat labels_image;
  cv::Mat stats;
  cv::Mat centroids;
  const int n_components = cv::connectedComponentsWithStats(
      img_labels, labels_image, stats, centroids, 8);

  for (int i = 1; i < n_components; ++i) {
    const int area = stats.at<std::int32_t>(i, cv::CC_STAT_AREA);
    if (area < min_area)
      continue;

    const double cx = centroids.at<double>(i, 0);
    const double cy = centroids.at<double>(i, 1);

    Arrow *best_arrow = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (Arrow &arrow : arrows) {
      for (const cv::Point &p : arrow.chemin) {
        const double d = (p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy);
        if (d < best_distance) {
          best_distance = d;
          best_arrow = &arrow;
        }
      }
    }

    if (best_arrow != nullptr) {
      const cv::Rect bbox(stats.at<std::int32_t>(i, cv::CC_STAT_LEFT),
                          stats.at<std::int32_t>(i, cv::CC_STAT_TOP),
                          stats.at<std::int32_t>(i, cv::CC_STAT_WIDTH),
                          stats.at<std::int32_t>(i, cv::CC_STAT_HEIGHT));
      ArrowLabel label;
      label.centroid = cv::Point2d(cx, cy);
      label.bbox = bbox;
      label.crop = extract_label_crop(img_labels, bbox, 2);
      best_arrow->labels.push_back(std::move(label));
    }
  }
}

} // namespace pyplus
