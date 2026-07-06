// Tests unitaires du kNN sur un petit dataset 2D construit a la main.

#include <iostream>
#include <optional>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/model_io.hh"

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

// Deux amas bien separes : "a" autour de (0,0), "b" autour de (10,10).
pyplus::Model make_model()
{
    pyplus::Model m;
    m.d = 2;
    m.classes = {"a", "b"};
    const std::vector<std::pair<float, float>> pts_a = {{0, 0}, {0, 1}, {1, 0}};
    const std::vector<std::pair<float, float>> pts_b = {{10, 10}, {10, 11}, {11, 10}};
    for (auto [px, py] : pts_a)
    {
        m.x.insert(m.x.end(), {px, py});
        m.y.push_back(0);
    }
    for (auto [px, py] : pts_b)
    {
        m.x.insert(m.x.end(), {px, py});
        m.y.push_back(1);
    }
    m.n = static_cast<int>(m.y.size());
    return m;
}

} // namespace

int main()
{
    // vote simple : majorite des k voisins
    pyplus::KNNClassifier knn(3);
    knn.fit(make_model());
    CHECK(knn.predict({0.2f, 0.2f}) == "a");
    CHECK(knn.predict({10.5f, 10.5f}) == "b");

    // k > N : tous les points votent, "a" et "b" a egalite 3-3
    // -> premier label dans l'ordre trie (comme np.unique + argmax)
    pyplus::KNNClassifier knn_all(100);
    knn_all.fit(make_model());
    CHECK(knn_all.predict({5.0f, 5.0f}) == "a");

    // vote pondere : un point d'entrainement exact domine (distance ~0)
    pyplus::KNNClassifier weighted(3, true);
    weighted.fit(make_model());
    CHECK(weighted.predict({0.0f, 0.0f}) == "a");
    CHECK(weighted.predict({10.0f, 11.0f}) == "b");
    // proche de l'amas "b" : les poids 1/d^2 doivent l'emporter
    CHECK(weighted.predict({9.0f, 9.0f}) == "b");

    // rejet : distance min au-dela du seuil -> nullopt
    pyplus::KNNClassifier reject(3, true, 5.0f);
    reject.fit(make_model());
    CHECK(reject.predict({100.0f, 100.0f}) == std::nullopt);
    CHECK(reject.predict({0.5f, 0.5f}) == "a");

    std::cout << "classifieur OK\n";
    return 0;
}
