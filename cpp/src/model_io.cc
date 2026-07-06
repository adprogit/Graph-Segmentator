#include "pyplus/model_io.hh"

#include <cstdint>
#include <fstream>
#include <stdexcept>

// Le format est little-endian ; on suppose un hote little-endian
// (x86-64, arm64), comme le prototype numpy.

namespace pyplus
{

namespace
{

std::int32_t read_i32(std::ifstream& f, const std::string& path)
{
    std::int32_t v = 0;
    if (!f.read(reinterpret_cast<char*>(&v), sizeof(v)))
        throw std::runtime_error("modele tronque : " + path);
    return v;
}

void write_i32(std::ofstream& f, std::int32_t v)
{
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

} // namespace

Model load_model(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("modele introuvable : " + path);

    Model model;
    model.n = read_i32(f, path);
    model.d = read_i32(f, path);
    const std::int32_t n_classes = read_i32(f, path);
    if (model.n <= 0 || model.d <= 0 || n_classes <= 0)
        throw std::runtime_error("en-tete de modele invalide : " + path);

    model.x.resize(static_cast<std::size_t>(model.n) * model.d);
    if (!f.read(reinterpret_cast<char*>(model.x.data()),
                static_cast<std::streamsize>(model.x.size() * sizeof(float))))
        throw std::runtime_error("modele tronque (X) : " + path);

    std::vector<std::int32_t> y_idx(static_cast<std::size_t>(model.n));
    if (!f.read(reinterpret_cast<char*>(y_idx.data()),
                static_cast<std::streamsize>(y_idx.size() * sizeof(std::int32_t))))
        throw std::runtime_error("modele tronque (y) : " + path);
    model.y.assign(y_idx.begin(), y_idx.end());

    for (std::int32_t c = 0; c < n_classes; ++c)
    {
        const std::int32_t len = read_i32(f, path);
        if (len < 0)
            throw std::runtime_error("label de longueur invalide : " + path);
        std::string label(static_cast<std::size_t>(len), '\0');
        if (!f.read(label.data(), len))
            throw std::runtime_error("modele tronque (labels) : " + path);
        model.classes.push_back(label);
    }

    for (int idx : model.y)
        if (idx < 0 || idx >= n_classes)
            throw std::runtime_error("index de classe hors bornes : " + path);

    return model;
}

void save_model(const std::string& path, const Model& model)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("impossible d'ecrire : " + path);

    write_i32(f, model.n);
    write_i32(f, model.d);
    write_i32(f, static_cast<std::int32_t>(model.classes.size()));
    f.write(reinterpret_cast<const char*>(model.x.data()),
            static_cast<std::streamsize>(model.x.size() * sizeof(float)));
    for (int idx : model.y)
        write_i32(f, idx);
    for (const std::string& label : model.classes)
    {
        write_i32(f, static_cast<std::int32_t>(label.size()));
        f.write(label.data(), static_cast<std::streamsize>(label.size()));
    }
    if (!f)
        throw std::runtime_error("ecriture incomplete : " + path);
}

} // namespace pyplus
