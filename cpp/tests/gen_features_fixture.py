"""
Genere le fixture de validation croisee des features Python -> C++.

Rend les glyphes de l'alphabet via Graphviz (comme l'entrainement), les
augmente (seed fixe), puis ecrit pour chaque crop brut : le crop, sa
version normalisee (normalize_crop) et son vecteur HOG (compute_hog)
attendus, dans cpp/tests/fixtures/features_expected.txt :

    ligne 1 : M
    puis M blocs :
        h w
        h lignes de w caracteres '0'/'1' (1 = trait = 255)   crop brut
        32 lignes de 32 caracteres                            crop normalise
        144 floats sur une ligne                              HOG attendu

Le test C++ test_features_cross recalcule normalize_crop + compute_hog
sur le crop brut et compare.

Usage (depuis la racine du depot, graphviz requis) :
    uv run --project python cpp/tests/gen_features_fixture.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import numpy as np

from features import compute_hog, normalize_crop
from training import augment, build_char_base

ALPHABET = "vbcndz"
N_PER_CHAR = 5
OUT = ROOT / "cpp" / "tests" / "fixtures" / "features_expected.txt"


def write_binary_image(f, img):
    f.write(f"{img.shape[0]} {img.shape[1]}\n")
    for row in img:
        f.write("".join("1" if v > 0 else "0" for v in row) + "\n")


def main():
    rng = np.random.default_rng(0)
    crops = []
    for char in ALPHABET:
        bases = build_char_base(char)
        if not bases:
            sys.exit(f"rendu Graphviz echoue pour '{char}' (dot installe ?)")
        for _ in range(N_PER_CHAR):
            base = bases[rng.integers(len(bases))]
            crops.append(augment(base, rng))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"{len(crops)}\n")
        for crop in crops:
            norm = normalize_crop(crop)
            hog = compute_hog(norm)
            write_binary_image(f, crop)
            write_binary_image(f, norm)
            f.write(" ".join(f"{v:.9g}" for v in hog) + "\n")

    print(f"{len(crops)} crops ecrits dans {OUT}")


if __name__ == "__main__":
    main()
