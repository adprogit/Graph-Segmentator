"""
Genere le fixture de validation croisee de la segmentation Python -> C++.

Genere quelques automates par niveau (Graphviz), copie les PNG dans
cpp/tests/fixtures/seg/, execute le pipeline complet du prototype
(segment_automaton + kNN) sur chaque image et ecrit la structure attendue
dans cpp/tests/fixtures/seg/expected_structure.txt :

    ligne 1 : M (nombre d'images)
    puis M blocs :
        nom_du_png
        n_states initial          (initial = -1 si absent)
        n_states valeurs 0/1      (flags acceptants)
        n_arrows
        n_arrows lignes : src dst n_syms sym1 sym2 ...

Le test C++ test_seg_cross re-execute le pipeline sur les memes PNG et
compare la structure (etats, initial, acceptants, arcs et symboles).

Usage (depuis la racine du depot, graphviz requis) :
    uv run --project python cpp/tests/gen_seg_fixture.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from automaton_generator import generate_corpus
from main import load_classifier, segment_automaton

N_PER_LEVEL = 3
OUT_DIR = ROOT / "cpp" / "tests" / "fixtures" / "seg"


def main():
    classifier = load_classifier(str(ROOT / "data" / "knn_model.bin"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        generate_corpus(tmp, n_per_level=N_PER_LEVEL)
        images = sorted(Path(tmp).glob("*/dfa_*.png"))
        names = []
        for png in images:
            name = f"{png.parent.name}_{png.name}"
            shutil.copy(png, OUT_DIR / name)
            names.append(name)

    with open(OUT_DIR / "expected_structure.txt", "w") as f:
        f.write(f"{len(names)}\n")
        for name in names:
            result = segment_automaton(str(OUT_DIR / name),
                                       classifier=classifier)
            states = result["states"]
            arrows = result["arrows"]
            initial = result["initial"]
            f.write(f"{name}\n")
            f.write(f"{len(states)} {initial if initial is not None else -1}\n")
            f.write(" ".join("1" if s.get("accepting") else "0"
                             for s in states) + "\n")
            f.write(f"{len(arrows)}\n")
            for edge in arrows:
                syms = [lbl["symbol"] for lbl in edge["labels"]
                        if lbl["symbol"] is not None]
                f.write(f"{edge['source']} {edge['dest']} {len(syms)} "
                        + " ".join(syms) + "\n")
            print(f"{name}: {len(states)} etats, {len(arrows)} arcs")

    print(f"\n{len(names)} images ecrites dans {OUT_DIR}")


if __name__ == "__main__":
    main()
