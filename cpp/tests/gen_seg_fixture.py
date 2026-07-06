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

Ecrit aussi les tables reconstruites (result_to_table) dans
expected_tables.txt :

    ligne 1 : M
    puis M blocs : nom_du_png, nombre de lignes de la table, la table

Les tests C++ test_seg_cross et test_table_cross re-executent le pipeline
sur les memes PNG et comparent structure et table.

Usage (depuis la racine du depot, graphviz requis) :
    uv run --project python cpp/tests/gen_seg_fixture.py
"""

import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from automaton_generator import generate_corpus
from export_table import result_to_table
from main import load_classifier, segment_automaton

N_PER_LEVEL = 3
OUT_DIR = ROOT / "cpp" / "tests" / "fixtures" / "seg"


def main():
    classifier = load_classifier(str(ROOT / "data" / "knn_model.bin"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # corpus reproductible (le generateur utilise `random`) ; certains tirages
    # produisent des images sur lesquelles le prototype lui-meme plante
    random.seed(42)
    with tempfile.TemporaryDirectory() as tmp:
        generate_corpus(tmp, n_per_level=N_PER_LEVEL)
        images = sorted(Path(tmp).glob("*/dfa_*.png"))
        names = []
        for png in images:
            name = f"{png.parent.name}_{png.name}"
            shutil.copy(png, OUT_DIR / name)
            names.append(name)

    # le prototype plante sur certaines images (0 etat detecte ->
    # analyze_tip dereference None) : on les ecarte du fixture
    results = {}
    for name in list(names):
        try:
            results[name] = segment_automaton(str(OUT_DIR / name),
                                              classifier=classifier)
        except Exception as e:
            print(f"[!] {name}: le prototype plante ({type(e).__name__}), "
                  f"image ecartee")
            (OUT_DIR / name).unlink()
            names.remove(name)

    with open(OUT_DIR / "expected_tables.txt", "w") as f:
        f.write(f"{len(names)}\n")
        for name in names:
            table = result_to_table(results[name])
            lines = table.splitlines()
            f.write(f"{name}\n{len(lines)}\n")
            f.write("\n".join(lines) + "\n")

    with open(OUT_DIR / "expected_structure.txt", "w") as f:
        f.write(f"{len(names)}\n")
        for name in names:
            result = results[name]
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
