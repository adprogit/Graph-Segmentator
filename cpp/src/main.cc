// Point d'entree du portage C++ (equivalent de python/src/main.py) :
// segmentation + reconnaissance d'une image d'automate.
//
//     pyplus <image> [--model data/knn_model.bin]
//
// Etat du portage, dans l'ordre du pipeline :
//     [x] model_io  [x] classifier  [x] features  [x] segmentation
//     [ ] export_table

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/model_io.hh"
#include "pyplus/pipeline.hh"

namespace
{

// Affiche un resume textuel de la segmentation (cf. print_summary Python).
void print_summary(const pyplus::AutomatonResult& result)
{
    const std::vector<pyplus::State>& states = result.states;

    std::cout << states.size() << " etat(s) :\n";
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        std::string tag;
        if (result.initial && *result.initial == static_cast<int>(i))
            tag += "initial";
        if (states[i].accepting)
            tag += (tag.empty() ? "" : ", ") + std::string("acceptant");
        std::cout << "  s" << i << ": centre=("
                  << std::lround(states[i].center_x) << ", "
                  << std::lround(states[i].center_y) << ")"
                  << (tag.empty() ? "" : " [" + tag + "]") << "\n";
    }

    std::cout << "\n" << result.arrows.size()
              << " transition(s) detectee(s) :\n";
    for (const pyplus::Arrow& arrow : result.arrows)
    {
        std::string symbols;
        std::size_t n_labels = 0;
        for (const pyplus::ArrowLabel& label : arrow.labels)
        {
            ++n_labels;
            if (label.symbol)
                symbols += (symbols.empty() ? "" : ", ") + *label.symbol;
        }
        if (!symbols.empty())
            std::cout << "  s" << arrow.source << " --" << symbols << "--> s"
                      << arrow.dest << "\n";
        else
            std::cout << "  s" << arrow.source << " -> s" << arrow.dest
                      << "  (" << n_labels
                      << " etiquette(s) non reconnue(s))\n";
    }

    std::cout << "\nMatrice d'adjacence :\n     ";
    for (std::size_t j = 0; j < states.size(); ++j)
        std::cout << (j > 0 ? "  " : "") << "s" << j;
    std::cout << "\n";
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        std::cout << "s" << i << ":  ";
        for (std::size_t j = 0; j < states.size(); ++j)
            std::cout << (j > 0 ? "   " : "")
                      << (result.matrix[i][j] >= 0 ? "1" : "0");
        std::cout << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::string image_path;
    std::string model_path = "data/knn_model.bin";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (image_path.empty())
            image_path = arg;
    }
    if (image_path.empty())
    {
        std::cerr << "usage: " << argv[0]
                  << " <image> [--model data/knn_model.bin]\n";
        return 1;
    }

    pyplus::KNNClassifier classifier(3, true);
    bool has_classifier = true;
    try
    {
        classifier.fit(pyplus::load_model(model_path));
    }
    catch (const std::exception& e)
    {
        has_classifier = false;
        std::cerr << "[!] " << e.what()
                  << " : segmentation seule, sans reconnaissance des "
                     "symboles.\n";
    }

    try
    {
        const pyplus::AutomatonResult result = pyplus::segment_automaton(
            image_path, has_classifier ? &classifier : nullptr);
        print_summary(result);
    }
    catch (const std::exception& e)
    {
        std::cerr << "erreur : " << e.what() << "\n";
        return 1;
    }
    return 0;
}
