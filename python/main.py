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
from training import LETTERS, DIGITS, train_and_save
from main import load_classifier              # src/main.py
from batch_eval import run_batch, print_table, save_failures

DATA = Path(__file__).resolve().parents[1] / "data"

CORPUS_DIR = str(DATA / "base_automata")
LETTERS_PATH = str(DATA / "knn_letters.bin")
DIGITS_PATH = str(DATA / "knn_digits.bin")
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
    """cf. training.__main__ (26 lettres + 10 chiffres, 400 exemples/caractere)."""
    print("\n=== 2. Entrainement des classifieurs ===")
    print("--- Modele lettres (a-z) ---")
    train_and_save(LETTERS, LETTERS_PATH)
    print("\n--- Modele chiffres (0-9) ---")
    train_and_save(DIGITS, DIGITS_PATH)


def step_eval():
    """cf. batch_eval.__main__ (corpus=base_automata, modeles=knn_*.bin)."""
    print("\n=== 3. Evaluation batch ===")
    classifier = None
    digit_classifier = None
    try:
        classifier = load_classifier(LETTERS_PATH)
    except FileNotFoundError:
        print(f"[!] modele {LETTERS_PATH} absent : evaluation structurelle seule.")
    try:
        digit_classifier = load_classifier(DIGITS_PATH)
    except FileNotFoundError:
        print(f"[!] modele {DIGITS_PATH} absent : noms d'etats non reconnus.")

    stats, failures = run_batch(CORPUS_DIR, classifier, digit_classifier,
                                global_threshold=0.9)
    print_table(stats)
    save_failures(failures, FAILURES_PATH)


def main():
    step_generate()
    step_train()
    step_eval()


if __name__ == "__main__":
    main()
