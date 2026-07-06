// Point d'entree du portage C++ (equivalent vise : python/src/main.py).
//
//     pyplus <image> [--model data/knn_model.bin]
//
// Etat du portage, dans l'ordre du pipeline :
//     [x] model_io     [x] classifier
//     [ ] segmentation [ ] features [ ] export_table

#include <exception>
#include <iostream>
#include <string>

#include "pyplus/classifier.hh"
#include "pyplus/model_io.hh"

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
    try
    {
        classifier.fit(pyplus::load_model(model_path));
        const pyplus::Model& m = classifier.model();
        std::cout << "modele " << model_path << " : " << m.n
                  << " exemples, dim " << m.d << ", classes";
        for (const std::string& c : m.classes)
            std::cout << " " << c;
        std::cout << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "[!] " << e.what()
                  << " : segmentation seule, sans reconnaissance\n";
    }

    std::cout << "segmentation non portee, image ignoree (" << image_path
              << ")\n";
    return 0;
}
