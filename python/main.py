"""
Pipeline complet, de bout en bout :

    1. supprime puis recree le corpus `base_automata/`
    2. genere le corpus            (automaton_generator)
    3. entraine le classifieur kNN (training)
    4. evalue le pipeline en batch (batch_eval)

Chaque etape reprend exactement ce que fait le `__main__` du module concerne.

    python main.py
"""

import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from automaton_generator import generate_corpus
from training import (
    generate_dataset, train_test_split, confusion_report, KNNClassifier,
)
from model_io import save_model
from main import load_classifier              # src/main.py
from batch_eval import run_batch, print_table, save_failures

DATA = Path(__file__).resolve().parents[1] / "data"

CORPUS_DIR = str(DATA / "base_automata")
MODEL_PATH = str(DATA / "knn_model.bin")
FAILURES_PATH = str(DATA / "failures.json")


def step_generate():
    """cf. automaton_generator.__main__ (n_per_level=50)."""
    print("\n=== 1. Generation du corpus ===")
    if Path(CORPUS_DIR).exists():
        shutil.rmtree(CORPUS_DIR)
    stats = generate_corpus(CORPUS_DIR, n_per_level=50)
    for level, info in stats.items():
        print(f"{level:7s} | {info['count']} DFA | "
              f"etats {info['states_range']} | "
              f"alphabet {info['alpha_range']}")


def step_train():
    """cf. training.__main__ (alphabet 'vbcndz', 400 exemples/caractere)."""
    print("\n=== 2. Entrainement du classifieur ===")
    ALPHABET = list("vbcndz")   # adapte a ton alphabet reel
    N_PER_CHAR = 400

    print("Generation du dataset...")
    X, y = generate_dataset(ALPHABET, n_per_char=N_PER_CHAR)
    print(f"  {len(X)} exemples, dim HOG = {X.shape[1]}")

    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.2)
    print(f"  train={len(X_tr)}  test={len(X_te)}")

    print("Entrainement kNN...")
    clf = KNNClassifier(k=3, weighted=True)
    clf.fit(X_tr, y_tr)

    print("Evaluation...")
    y_pred = clf.predict(X_te)
    cm, acc = confusion_report(y_te, y_pred, ALPHABET)

    print(f"\nAccuracy : {acc:.1%}\n")
    print("Matrice de confusion (lignes=vrai, colonnes=predit) :")
    print("      " + "  ".join(f"{c:>3}" for c in ALPHABET))
    for i, c in enumerate(ALPHABET):
        row = "  ".join(f"{cm[i, j]:>3}" for j in range(len(ALPHABET)))
        print(f"  {c} : {row}")

    save_model(MODEL_PATH, X, y)
    print(f"\nModele sauvegarde dans {MODEL_PATH} "
          f"({len(X)} exemples, dim {X.shape[1]}).")


def step_eval():
    """cf. batch_eval.__main__ (corpus=base_automata, modele=knn_model.bin)."""
    print("\n=== 3. Evaluation batch ===")
    classifier = None
    try:
        classifier = load_classifier(MODEL_PATH)
    except FileNotFoundError:
        print(f"[!] modele {MODEL_PATH} absent : evaluation structurelle seule.")

    stats, failures = run_batch(CORPUS_DIR, classifier, global_threshold=0.9)
    print_table(stats)
    save_failures(failures, FAILURES_PATH)


def main():
    step_generate()
    step_train()
    step_eval()


if __name__ == "__main__":
    main()
