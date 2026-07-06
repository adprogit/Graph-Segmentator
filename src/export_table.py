"""
Export du resultat de segmentation vers le format table, et comparaison
avec une table de reference.

Le format :
    #states / #initial / #accepting / #alphabet / #transitions
    transitions au format src:sym>dst
"""

from automaton_parser import Automaton, parse_table, to_table
from automaton_compare import compare, print_report


def result_to_table(result):
    """
    Convertit le dict de segment_automaton en une chaine au format table.

    Attend :
        result["states"]  : liste d'etats (avec flag "accepting")
        result["initial"] : index de l'etat initial (ou None)
        result["arrows"]  : aretes avec labels reconnus (label["symbol"])
    """
    states = result["states"]
    n = len(states)
    state_names = [f"s{i}" for i in range(n)]

    # alphabet + transitions, deduits des symboles reconnus
    alphabet = set()
    transitions = set()  # (src, sym, dst)
    for edge in result["arrows"]:
        src = f"s{edge['source']}"
        dst = f"s{edge['dest']}"
        for label in edge.get("labels", []):
            sym = label.get("symbol")
            if sym is None:
                continue  # non reconnu -> on n'invente pas
            alphabet.add(sym)
            transitions.add((src, sym, dst))

    initial = result["initial"]
    initial_name = f"s{initial}" if initial is not None else (
        state_names[0] if state_names else None)

    aut = Automaton(
        states=set(state_names),
        initial=initial_name,
        accepting={f"s{i}" for i, s in enumerate(states) if s.get("accepting")},
        alphabet=alphabet,
        transitions=transitions,
    )
    return to_table(aut)


def save_table(result, path):
    """Ecrit la table reconstruite sur disque."""
    table_str = result_to_table(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table_str)
    return path


def compare_with_reference(result_path, reference_path, verbose=True):
    """
    Compare la table reconstruite avec une table de reference.

    Returns:
        le dict de comparaison (scores, isomorphisme, diffs).
    """
    predicted = parse_table(result_path)
    reference = parse_table(reference_path)
    report = compare(reference, predicted)
    if verbose:
        print_report(report)
    return report


# ---------- test ----------
if __name__ == "__main__":
    # resultat simule (ce que produirait segment_automaton)
    fake_result = {
        "states": [
            {"center_x": 90, "center_y": 80, "accepting": False},
            {"center_x": 210, "center_y": 80, "accepting": True},
        ],
        "initial": 0,
        "arrows": [
            {"source": 0, "dest": 0, "labels": [{"symbol": "b"}]},
            {"source": 0, "dest": 1, "labels": [{"symbol": "d"}]},
            {"source": 1, "dest": 1, "labels": [{"symbol": "b"}]},
            {"source": 1, "dest": 0, "labels": [{"symbol": "d"}]},
        ],
    }

    table = result_to_table(fake_result)
    print("=== TABLE RECONSTRUITE ===")
    print(table)

    save_table(fake_result, "predicted.txt")

    # reference identique pour le test
    ref = """#states
s0
s1
#initial
s0
#accepting
s1
#alphabet
b
d
#transitions
s0:b>s0
s0:d>s1
s1:b>s1
s1:d>s0
"""
    with open("reference.txt", "w") as f:
        f.write(ref)

    print("=== COMPARAISON ===")
    compare_with_reference("predicted.txt", "reference.txt")
