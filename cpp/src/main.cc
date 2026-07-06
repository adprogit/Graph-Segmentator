// Point d'entree du portage C++ (equivalent vise : python/src/main.py).
//
//     pyplus <image> [--model data/knn_model.bin]
//
// Modules a porter depuis python/src/, dans l'ordre du pipeline :
//     segmentation -> features -> classifier -> model_io -> export_table

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0]
                  << " <image> [--model data/knn_model.bin]\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::cout << "squelette c++ : pipeline non porte, image ignoree ("
              << image_path << ")\n";
    return 0;
}
