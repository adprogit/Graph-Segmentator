// Validation croisee : le pipeline C++ complet (segmentation + kNN) doit
// reproduire la structure extraite par le prototype Python sur les memes
// images d'automates (fixture genere par gen_seg_fixture.py).
//
//     test_seg_cross <knn_model.bin> <fixtures/seg>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/model_io.hh"
#include "pyplus/pipeline.hh"

namespace
{

// Structure attendue, serialisee comme dans le fixture pour comparaison.
std::string describe(const pyplus::AutomatonResult& result)
{
    std::ostringstream out;
    out << result.states.size() << " "
        << (result.initial ? *result.initial : -1) << "\n";
    for (std::size_t i = 0; i < result.states.size(); ++i)
        out << (i > 0 ? " " : "") << (result.states[i].accepting ? 1 : 0);
    out << "\n" << result.arrows.size() << "\n";
    for (const pyplus::Arrow& edge : result.arrows)
    {
        std::vector<std::string> syms;
        for (const pyplus::ArrowLabel& label : edge.labels)
            if (label.symbol)
                syms.push_back(*label.symbol);
        out << edge.source << " " << edge.dest << " " << syms.size();
        for (const std::string& s : syms)
            out << " " << s;
        out << "\n";
    }
    return out.str();
}

std::string read_expected_block(std::ifstream& f)
{
    std::ostringstream out;
    std::string header_line;
    std::getline(f, header_line);  // "n_states initial"
    out << header_line << "\n";
    std::string accepting_line;
    std::getline(f, accepting_line);
    out << accepting_line << "\n";
    std::string n_arrows_line;
    std::getline(f, n_arrows_line);
    out << n_arrows_line << "\n";
    const int n_arrows = std::stoi(n_arrows_line);
    for (int i = 0; i < n_arrows; ++i)
    {
        std::string arrow_line;
        std::getline(f, arrow_line);
        // le generateur laisse une espace finale quand il n'y a aucun symbole
        while (!arrow_line.empty() && arrow_line.back() == ' ')
            arrow_line.pop_back();
        out << arrow_line << "\n";
    }
    return out.str();
}

std::string trim_trailing_spaces(const std::string& block)
{
    std::istringstream in(block);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line))
    {
        while (!line.empty() && line.back() == ' ')
            line.pop_back();
        out << line << "\n";
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " <knn_model.bin> <fixtures/seg>\n";
        return 2;
    }
    const std::string seg_dir = argv[2];

    pyplus::KNNClassifier classifier(3, true);
    classifier.fit(pyplus::load_model(argv[1]));

    std::ifstream fixture(seg_dir + "/expected_structure.txt");
    if (!fixture)
    {
        std::cerr << "fixture introuvable dans " << seg_dir
                  << " (generer avec gen_seg_fixture.py)\n";
        return 2;
    }

    int n_images = 0;
    fixture >> n_images;
    {
        std::string rest;
        std::getline(fixture, rest);
    }

    int n_mismatch = 0;
    for (int i = 0; i < n_images; ++i)
    {
        std::string name;
        std::getline(fixture, name);
        const std::string expected = read_expected_block(fixture);
        if (!fixture)
        {
            std::cerr << "fixture tronque a l'image " << i << "\n";
            return 2;
        }

        const pyplus::AutomatonResult result =
            pyplus::segment_automaton(seg_dir + "/" + name, &classifier);
        const std::string got = trim_trailing_spaces(describe(result));

        if (got != expected)
        {
            std::cerr << "=== " << name << " : divergence ===\n"
                      << "attendu :\n" << expected
                      << "obtenu :\n" << got << "\n";
            ++n_mismatch;
        }
        else
        {
            std::cout << name << " OK\n";
        }
    }

    if (n_mismatch > 0)
    {
        std::cerr << n_mismatch << "/" << n_images << " images divergentes\n";
        return 1;
    }
    std::cout << n_images << " images identiques a Python\n";
    return 0;
}
