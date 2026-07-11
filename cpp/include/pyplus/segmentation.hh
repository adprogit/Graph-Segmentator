// Segmentation d'une image d'automate, port ligne a ligne de
// python/src/segmentation.py : detection des etats (cercles), des tetes de
// fleche (triangles), suivi de trace des arcs, isolation des etiquettes.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace pyplus {

struct State {
  double center_x = 0.0;
  double center_y = 0.0;
  double radius = 0.0;
  double outer_radius = 0.0;
  bool accepting = false;
  int area = 0;
  cv::Mat name_crop;
  std::string name; // nom final (reconnu ou repli s{i}), vide avant attribution
};

struct Tip {
  cv::Point2d centroid;
  int area = 0;
  cv::Rect bbox;
  std::vector<cv::Point> pixels; // coordonnees globales, ordre ligne a ligne
};

struct ArrowLabel {
  cv::Point2d centroid;
  cv::Rect bbox;
  cv::Mat crop;
  std::optional<std::string> symbol;
};

struct Arrow {
  int source = -1;
  int dest = -1;
  std::vector<cv::Point> chemin;
  std::vector<ArrowLabel> labels;
};

struct AdjacencyResult {
  // matrix[src][dst] = index dans arrows, ou -1 (le prototype y met une
  // reference vers l'objet arete)
  std::vector<std::vector<int>> matrix;
  std::optional<int> initial;
  std::vector<Arrow> arrows;
};

// Charge en niveaux de gris (Mat vide si introuvable, comme cv2.imread).
cv::Mat load_image(const std::string &path);

// Binarise via Otsu : trait=255 (blanc), fond=0 (noir).
cv::Mat to_binary(const cv::Mat &gray);

// Detecte les etats : composantes du fond circulaires (l'interieur d'un
// cercle est un trou du fond). Le seuil d'aire est relatif a l'image mais
// plafonne : les cercles d'etats ont une taille a peu pres fixe (rendu
// Graphviz), alors que l'image grandit avec l'automate.
std::vector<State> segment_states(const cv::Mat &binary_image,
                                  double relative_min_area = 0.005,
                                  double min_area_cap = 500.0);

// Efface les etats (outer_radius, pour couvrir les double-cercles).
cv::Mat remove_states_from_img(const cv::Mat &binary_image,
                               const std::vector<State> &states,
                               int padding = 3);

// Marque acceptant tout etat dont un cercle residuel partage le centre.
void merge_states_and_acceptants(std::vector<State> &states_first_pass,
                                 const std::vector<State> &residual_circles,
                                 double center_tolerance = 10.0);

// Tetes de fleche : composantes non circulaires apres ouverture 3x3.
std::vector<Tip> find_triangles(const cv::Mat &img_cleaned);

// Pour chaque triangle : destination (etat le plus proche), suivi de trace
// vers la source, fallback geometrique, sinon fleche initiale.
AdjacencyResult build_adjacency_matrix(const std::vector<State> &states,
                                       const std::vector<Tip> &triangles,
                                       const cv::Mat &img_clean_final);

// Efface les composantes appartenant aux arcs (chemins + tetes) : il ne
// reste que les etiquettes.
cv::Mat isolate_labels(const cv::Mat &img_clean_final,
                       const std::vector<Tip> &tips,
                       const std::vector<Arrow> &arrows);

// Decoupe la lettre dans sa bbox, centre et normalise en target_size.
cv::Mat extract_label_crop(const cv::Mat &img_labels, const cv::Rect &bbox,
                           int padding = 2, int target_size = 32);

// Decoupe le nom d'un etat (name_crop) en caracteres : composantes
// connexes triees de gauche a droite, chacune extraite via son propre
// masque (le padding de extract_label_crop n'attrape ainsi pas les
// pixels du caractere voisin).
std::vector<cv::Mat> segment_name_characters(const cv::Mat &name_crop,
                                             int min_area = 5,
                                             int target_size = 32);

// Rattache chaque etiquette a l'arc dont le chemin passe au plus pres.
void assign_labels_to_arrows(const cv::Mat &img_labels,
                             std::vector<Arrow> &arrows, int min_area = 10);

} // namespace pyplus
