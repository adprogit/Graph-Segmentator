"""
Genere les fixtures de validation croisee Python -> C++ du kNN.

Pour chacun des deux modeles (data/knn_letters.bin, data/knn_digits.bin),
construit des vecteurs de requete deterministes (points d'entrainement
bruites + interpolations asymetriques entre paires), les predit avec le
kNN du prototype (k=3, pondere), et ecrit le tout dans
cpp/tests/fixtures/cross_predictions_{letters,digits}.txt :

    ligne 1 : M D
    puis M lignes : label_attendu f1 f2 ... fD

Le test C++ test_cross_python recharge le meme modele et doit reproduire
chaque prediction.

Usage (depuis la racine du depot) :
    uv run --project python cpp/tests/gen_cross_fixture.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import numpy as np

from classifier import KNNClassifier
from model_io import load_model

FIXTURES = ROOT / "cpp" / "tests" / "fixtures"
MODELS = {
    "letters": ROOT / "data" / "knn_letters.bin",
    "digits": ROOT / "data" / "knn_digits.bin",
}


def generate_one(model_path, out_path):
    X, y = load_model(str(model_path))
    clf = KNNClassifier(k=3, weighted=True)
    clf.fit(X, y)

    rng = np.random.default_rng(0)
    noisy = X[rng.choice(len(X), size=40, replace=False)]
    noisy = (noisy + rng.normal(0, 0.05, size=noisy.shape)).astype(np.float32)
    # interpolation asymetrique (pas 0.5 : le milieu exact de deux points
    # est un cas d'egalite parfaite, dont l'issue depend des arrondis)
    pairs = rng.choice(len(X), size=(20, 2), replace=False)
    interp = (0.7 * X[pairs[:, 0]] + 0.3 * X[pairs[:, 1]]).astype(np.float32)
    queries = np.vstack([noisy, interp])

    predictions = clf.predict(queries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{len(queries)} {queries.shape[1]}\n")
        for q, label in zip(queries, predictions):
            # .9g : round-trip exact d'un float32 en texte
            coords = " ".join(f"{v:.9g}" for v in q)
            f.write(f"{label} {coords}\n")

    print(f"{len(queries)} predictions ecrites dans {out_path}")


def main():
    for name, model_path in MODELS.items():
        generate_one(model_path, FIXTURES / f"cross_predictions_{name}.txt")


if __name__ == "__main__":
    main()
