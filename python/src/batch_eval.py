"""
Evaluation batch : parcourt un corpus d'images d'automates, reconstruit
chaque table via le pipeline, la compare a la verite terrain, agrege les
scores par niveau et exporte les echecs pour inspection.

Arborescence attendue :
    corpus_dir/
        simple_dfa/  dfa_000.png dfa_000.txt ...
        medium_dfa/  ...
        hard_dfa/    ...

Usage :
    python batch_eval.py base_automata --letters knn_letters.bin --digits knn_digits.bin
    python batch_eval.py base_automata --failures failures.json --csv scores.csv

    # sauvegarder aussi les tables predites (miroir du corpus)
    python batch_eval.py base_automata --out pred_py

    # scorer des tables deja produites (ex. par le batch C++), sans pipeline
    python batch_eval.py base_automata --pred pred_cpp
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path

from main import segment_automaton, load_classifier
from export_table import result_to_table
from automaton_parser import parse_table
from automaton_compare import compare

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# =============================================================================
# Evaluation unitaire
# =============================================================================




def evaluate_one(image_path, ref_path, classifier, digit_classifier=None,
                 out_path=None):
    """
    Reconstruit une table depuis une image et la compare a sa reference.
    Le chemin de l'image est toujours conserve dans le rapport (pour debug).
    Si out_path est fourni, la table predite y est aussi ecrite (meme
    contrat que le batch C++ : pas de fichier en cas d'erreur).
    """
    try:
        result = segment_automaton(image_path, classifier=classifier,
                                   digit_classifier=digit_classifier)
        table_str = result_to_table(result)
    except Exception as exc:
        return {"error": str(exc), "path": image_path}

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table_str)

    predicted = parse_table(table_str, is_path=False)
    reference = parse_table(ref_path)
    report = compare(reference, predicted)
    report["path"] = image_path
    return report


def evaluate_pred(pred_path, ref_path, image_path):
    """
    Compare une table predite deja sur disque (ex. sortie du batch C++)
    a sa reference. Une table absente ou illisible est un echec dur,
    comme une exception de la pipeline en mode direct.
    """
    try:
        predicted = parse_table(pred_path)
    except FileNotFoundError:
        return {"error": f"table predite absente : {pred_path}",
                "path": image_path}
    except Exception as exc:
        return {"error": str(exc), "path": image_path}

    reference = parse_table(ref_path)
    report = compare(reference, predicted)
    report["path"] = image_path
    return report


# deux familles cote a cote : scores bruts (les noms d'etats comptent,
# mesure de la reconnaissance des noms) et scores alignes (structure,
# invariante au nommage)
DISPLAY_COLS = ["states", "alphabet", "transitions", "global",
                "aligned_transitions", "global_aligned",
                "exact_match_names", "exact_match"]

COL_HEADERS = {
    "states": "states", "alphabet": "alphabet", "accepting": "accept",
    "initial": "initial", "transitions": "trans",
    "aligned_accepting": "al_acc", "aligned_initial": "al_init",
    "aligned_transitions": "al_trans",
    "global": "global", "global_aligned": "g_align",
    "exact_match": "exact", "exact_match_names": "ex_names",
    "isomorphic": "iso",
}
 








# =============================================================================
# Agregation
# =============================================================================

# metriques du dict `scores` (Jaccard brut par section)
SECTION_KEYS = ["states", "alphabet", "accepting", "initial", "transitions"]
# metriques au niveau racine du rapport (brutes + alignees name-invariant)
ROOT_KEYS = ["global", "global_aligned",
             "aligned_accepting", "aligned_initial", "aligned_transitions"]


def aggregate(reports):
    """Moyennes par section + taux, en ignorant les rapports en erreur."""
    valid = [r for r in reports if "error" not in r]
    n_error = len(reports) - len(valid)
    if not valid:
        return {"n": len(reports), "n_error": n_error}

    stats = {"n": len(reports), "n_error": n_error}

    for key in SECTION_KEYS:
        stats[key] = sum(r["scores"][key] for r in valid) / len(valid)

    for key in ROOT_KEYS:
        # certaines cles peuvent manquer si compare() ne les fournit pas
        vals = [r[key] for r in valid if key in r]
        stats[key] = sum(vals) / len(vals) if vals else float("nan")

    # exact_match : structure parfaite (name-invariant) ;
    # exact_match_names : table litteralement parfaite, noms compris
    stats["exact_match"] = sum(1 for r in valid if r["global_aligned"] >= 0.999) / len(valid)
    stats["exact_match_names"] = sum(1 for r in valid if r["global"] >= 0.999) / len(valid)
    stats["isomorphic"] = sum(1 for r in valid if r.get("isomorphic")) / len(valid)
    return stats


def collect_failures(reports, global_threshold=0.9):
    """
    Separe les echecs durs (exception) des echecs mous (score faible),
    avec le detail des transitions divergentes pour le debug.
    """
    hard = [r["path"] for r in reports if "error" in r]

    soft = []
    for r in reports:
        if "error" in r or r["global_aligned"] >= global_threshold:
            continue
        soft.append({
            "path": r["path"],
            "global_aligned": round(r["global_aligned"], 4),
            "global": round(r["global"], 4),
            "aligned_transitions": round(r.get("aligned_transitions", float("nan")), 4),
            "isomorphic": r.get("isomorphic"),
            "missing": [f"{s}:{sym}>{d}"
                        for (s, sym, d) in r.get("missing_transitions", [])],
            "extra": [f"{s}:{sym}>{d}"
                      for (s, sym, d) in r.get("extra_transitions", [])],
        })
    soft.sort(key=lambda x: x["global_aligned"])  # du pire au moins pire
    return hard, soft


# =============================================================================
# Batch complet
# =============================================================================

def run_batch(corpus_dir, classifier, digit_classifier=None,
              global_threshold=0.9,
              levels=("simple_dfa", "medium_dfa", "hard_dfa"),
              pred_dir=None, out_dir=None):
    """
    Evalue chaque niveau. Retourne :
        stats    : {level: stats_agregees}
        failures : {level: {"hard": [...], "soft": [...]}}

    Deux modes :
        pred_dir : scoring seul, lit les tables predites dans pred_dir
                   (arborescence miroir du corpus, ex. sortie du batch C++) ;
        sinon    : inference via la pipeline, avec ecriture des tables
                   predites dans out_dir si fourni.
    """
    stats = {}
    failures = {}

    for level in levels:
        level_dir = os.path.join(corpus_dir, level)
        images = sorted(glob.glob(os.path.join(level_dir, "*.png")))
        if not images:
            continue

        reports = []
        for img_path in images:
            ref_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(ref_path):
                continue
            table_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            if pred_dir is not None:
                pred_path = os.path.join(pred_dir, level, table_name)
                reports.append(evaluate_pred(pred_path, ref_path, img_path))
            else:
                out_path = (os.path.join(out_dir, level, table_name)
                            if out_dir is not None else None)
                reports.append(evaluate_one(img_path, ref_path, classifier,
                                            digit_classifier,
                                            out_path=out_path))

        stats[level] = aggregate(reports)
        hard, soft = collect_failures(reports, global_threshold)
        failures[level] = {"hard": hard, "soft": soft}

    return stats, failures


# =============================================================================
# Sorties
# =============================================================================

def print_table(stats):
    header = f"{'niveau':<12} {'n':>4} {'err':>4}  " + \
             "  ".join(f"{COL_HEADERS[c]:>8}" for c in DISPLAY_COLS)
    print(header)
    print("-" * len(header))
    for level, s in stats.items():
        if "global" not in s:
            print(f"{level:<12} {s['n']:>4} {s['n_error']:>4}  (aucun valide)")
            continue
        row = f"{level:<12} {s['n']:>4} {s['n_error']:>4}  "
        cells = []
        for c in DISPLAY_COLS:
            v = s.get(c, float("nan"))
            cells.append("     n/a" if v != v else f"{v:>7.1%}")  # v!=v => NaN
        row += "  ".join(cells)
        print(row)


def save_failures(failures, out_path):
    """Exporte les echecs en JSON (horodate) pour inspection / regression."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "levels": failures,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    n_hard = sum(len(v["hard"]) for v in failures.values())
    n_soft = sum(len(v["soft"]) for v in failures.values())
    print(f"\nEchecs sauvegardes dans {out_path} "
          f"({n_hard} durs, {n_soft} mous).")


def save_scores_csv(stats, out_path):
    """Exporte les scores agreges en CSV (pour suivi entre runs)."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "n", "n_error"] + DISPLAY_COLS)
        for level, s in stats.items():
            if "global" not in s:
                writer.writerow([level, s["n"], s["n_error"]] + [""] * len(DISPLAY_COLS))
                continue
            row = [level, s["n"], s["n_error"]]
            row += [round(s.get(c, float("nan")), 4) for c in DISPLAY_COLS]
            writer.writerow(row)
    print(f"Scores sauvegardes dans {out_path}.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluation batch du pipeline.")
    parser.add_argument("corpus", help="dossier du corpus (contient simple_dfa/...)")
    parser.add_argument("--letters", default=str(DATA_DIR / "knn_letters.bin"),
                        help="modele kNN lettres (defaut: data/knn_letters.bin)")
    parser.add_argument("--digits", default=str(DATA_DIR / "knn_digits.bin"),
                        help="modele kNN chiffres (defaut: data/knn_digits.bin)")
    parser.add_argument("--failures", default=str(DATA_DIR / "failures.json"),
                        help="fichier JSON des echecs (defaut: data/failures.json)")
    parser.add_argument("--csv", default=None,
                        help="exporte aussi les scores agreges en CSV")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="seuil global_aligned sous lequel un cas est un echec mou")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pred", default=None,
                      help="score des tables deja predites (arborescence "
                           "miroir du corpus, ex. sortie du batch C++), "
                           "sans executer la pipeline")
    mode.add_argument("--out", default=None,
                      help="ecrit aussi les tables predites dans ce dossier "
                           "(meme format que le batch C++)")
    args = parser.parse_args()

    classifier = None
    digit_classifier = None
    if args.pred is None:
        try:
            classifier = load_classifier(args.letters)
        except FileNotFoundError:
            print(f"[!] modele {args.letters} absent : evaluation structurelle seule.")
        try:
            digit_classifier = load_classifier(args.digits)
        except FileNotFoundError:
            print(f"[!] modele {args.digits} absent : noms d'etats non reconnus.")

    stats, failures = run_batch(args.corpus, classifier, digit_classifier,
                                global_threshold=args.threshold,
                                pred_dir=args.pred, out_dir=args.out)
    print_table(stats)
    save_failures(failures, args.failures)
    if args.csv:
        save_scores_csv(stats, args.csv)


if __name__ == "__main__":
    main()
