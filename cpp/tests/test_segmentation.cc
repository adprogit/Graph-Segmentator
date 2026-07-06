// Tests unitaires de la segmentation sur un automate synthetique dessine
// a la main : deux etats (le second acceptant), un arc 0 -> 1, une fleche
// initiale sur l'etat 0.

#include <iostream>

#include <opencv2/imgproc.hpp>

#include "pyplus/segmentation.hh"

#define CHECK(cond)                                                       \
    do                                                                    \
    {                                                                     \
        if (!(cond))                                                      \
        {                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__                      \
                      << " echec : " #cond "\n";                          \
            return 1;                                                     \
        }                                                                 \
    } while (0)

namespace
{

// Image "papier" : fond blanc (255), trait noir (0), comme un rendu Graphviz.
cv::Mat draw_automaton()
{
    cv::Mat gray(120, 260, CV_8U, cv::Scalar(255));
    // etat 0 et etat 1 (double cercle = acceptant)
    cv::circle(gray, {60, 60}, 20, 0, 2);
    cv::circle(gray, {200, 60}, 20, 0, 2);
    cv::circle(gray, {200, 60}, 26, 0, 2);
    // arc 0 -> 1 : ligne fine + tete de fleche pleine. La ligne doit etre
    // fine (1 px) pour disparaitre a l'ouverture 3x3, comme un trait
    // Graphviz : sinon ligne+tete forment un seul blob "tete" geant et la
    // direction de l'arc s'inverse (comportement identique du prototype).
    cv::line(gray, {80, 60}, {160, 60}, 0, 1);
    const cv::Point head[] = {{170, 60}, {160, 55}, {160, 65}};
    cv::fillConvexPoly(gray, head, 3, 0);
    // fleche initiale -> etat 0 (tete seule, pas de trace en amont)
    const cv::Point init_head[] = {{38, 60}, {28, 54}, {28, 66}};
    cv::fillConvexPoly(gray, init_head, 3, 0);
    return gray;
}

} // namespace

int main()
{
    const cv::Mat gray = draw_automaton();

    // binarisation : trait=255, fond=0
    const cv::Mat binary = pyplus::to_binary(gray);
    CHECK(binary.at<std::uint8_t>(0, 0) == 0);        // fond
    CHECK(binary.at<std::uint8_t>(60, 100) == 255);   // trait de l'arc

    // deux etats, centres et rayons coherents
    std::vector<pyplus::State> states = pyplus::segment_states(binary);
    CHECK(states.size() == 2);
    for (const pyplus::State& s : states)
    {
        CHECK(std::abs(s.center_y - 60.0) < 3.0);
        CHECK(s.radius > 15.0 && s.radius < 25.0);
    }

    // second passage : le double cercle de l'etat acceptant
    const cv::Mat without_inner = pyplus::remove_states_from_img(binary, states);
    const std::vector<pyplus::State> residual =
        pyplus::segment_states(without_inner);
    pyplus::merge_states_and_acceptants(states, residual);
    const int accepting_index = states[0].center_x > states[1].center_x ? 0 : 1;
    CHECK(states[static_cast<std::size_t>(accepting_index)].accepting);
    CHECK(!states[static_cast<std::size_t>(1 - accepting_index)].accepting);

    // apres effacement des etats il ne reste que l'arc et les tetes
    const cv::Mat img_clean = pyplus::remove_states_from_img(binary, states);
    const std::vector<pyplus::Tip> tips = pyplus::find_triangles(img_clean);
    CHECK(tips.size() == 2);  // tete de l'arc + fleche initiale

    // matrice d'adjacence : un arc source -> dest, fleche initiale sur 0
    const pyplus::AdjacencyResult adjacency =
        pyplus::build_adjacency_matrix(states, tips, img_clean);
    CHECK(adjacency.arrows.size() == 1);
    const pyplus::Arrow& edge = adjacency.arrows[0];
    CHECK(edge.source == 1 - accepting_index);
    CHECK(edge.dest == accepting_index);
    CHECK(!edge.chemin.empty());
    CHECK(adjacency.matrix[static_cast<std::size_t>(edge.source)]
                          [static_cast<std::size_t>(edge.dest)] == 0);
    CHECK(adjacency.initial.has_value());
    CHECK(*adjacency.initial == 1 - accepting_index);

    // extract_label_crop : sortie 32x32 centree
    const cv::Mat crop =
        pyplus::extract_label_crop(binary, cv::Rect(50, 50, 20, 20));
    CHECK(crop.rows == 32 && crop.cols == 32);

    std::cout << "segmentation OK\n";
    return 0;
}
