// Round-trip save/load, equivalent du __main__ de python/src/model_io.py.

#include <cstdio>
#include <cstdlib>
#include <iostream>

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

int main()
{
    pyplus::Model model;
    model.n = 50;
    model.d = 144;
    model.classes = {"a", "b", "c", "d", "e"};
    std::srand(42);
    for (int i = 0; i < model.n; ++i)
    {
        model.y.push_back(i % 5);
        for (int j = 0; j < model.d; ++j)
            model.x.push_back(static_cast<float>(std::rand())
                              / static_cast<float>(RAND_MAX));
    }

    const std::string path = "test_model.bin";
    pyplus::save_model(path, model);
    const pyplus::Model loaded = pyplus::load_model(path);
    std::remove(path.c_str());

    CHECK(loaded.n == model.n);
    CHECK(loaded.d == model.d);
    CHECK(loaded.x == model.x);
    CHECK(loaded.y == model.y);
    CHECK(loaded.classes == model.classes);
    CHECK(loaded.label(7) == "c");

    bool threw = false;
    try
    {
        pyplus::load_model("inexistant.bin");
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    CHECK(threw);

    std::cout << "round-trip OK : " << loaded.n << " exemples, dim "
              << loaded.d << "\n";
    return 0;
}
