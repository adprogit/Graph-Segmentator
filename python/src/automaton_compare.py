
import networkx as nx
from networkx.algorithms import isomorphism

from automaton_parser import Automaton, parse_table
def weak_signature(state, aut):
    """
    Signature robuste aux transitions manquantes : uniquement initial/accepting.
    Sert a elaguer l'espace d'appariement sans casser les cas imparfaits
    (une transition ratee changerait les degres, donc on ne les met PAS ici).
    """
    return (state == aut.initial, state in aut.accepting)
 
def aligned_initial_score(mapping, pred, ref):
    """
    1.0 si l'etat initial predit, une fois mappe vers la reference,
    correspond a l'initial de reference. Robuste au renommage.
    """
    if pred.initial is None or ref.initial is None:
        return 1.0 if pred.initial == ref.initial else 0.0
    return 1.0 if mapping.get(pred.initial) == ref.initial else 0.0


def aligned_accepting_score(mapping, pred, ref):
    """
    Jaccard sur les etats acceptants, apres application du mapping aux
    acceptants predits. Robuste au renommage.
    """
    mapped = {mapping[s] for s in pred.accepting if s in mapping}
    return jaccard(mapped, ref.accepting) 

def count_matched_transitions(mapping, pred, ref):
    """Nb de transitions de pred qui, renommees via mapping, existent dans ref."""
    matched = 0
    for (s, sym, d) in pred.transitions:
        if s in mapping and d in mapping:
            if (mapping[s], sym, mapping[d]) in ref.transitions:
                matched += 1
    return matched
 
 
def find_best_mapping(pred, ref):
    """
    Cherche la bijection etats_pred -> etats_ref maximisant le nombre de
    transitions alignees. Backtracking avec elagage par signature faible.
 
    Retourne (mapping, nb_transitions_matchees).
    Complexite maitrisee pour de petits automates (<= ~10 etats).
    """
    pred_states = list(pred.states)
    ref_states = list(ref.states)
 
    sig_p = {s: weak_signature(s, pred) for s in pred_states}
    sig_r = {t: weak_signature(t, ref) for t in ref_states}
 
    candidates = {
        s: [t for t in ref_states if sig_r[t] == sig_p[s]]
        for s in pred_states
    }
    # etats les plus contraints d'abord -> elagage plus efficace
    order = sorted(pred_states, key=lambda s: len(candidates[s]))
 
    best = {"mapping": {}, "score": -1}
 
    def backtrack(idx, assigned, used):
        if idx == len(order):
            score = count_matched_transitions(assigned, pred, ref)
            if score > best["score"]:
                best["mapping"] = dict(assigned)
                best["score"] = score
            return
        s = order[idx]
        for t in candidates[s]:
            if t in used:
                continue
            assigned[s] = t
            used.add(t)
            backtrack(idx + 1, assigned, used)
            used.discard(t)
            del assigned[s]
        # laisser s non mappe (tailles differentes / etat parasite detecte)
        backtrack(idx + 1, assigned, used)
 
    backtrack(0, {}, set())
    return best["mapping"], best["score"]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def to_graph(aut: Automaton) -> nx.MultiDiGraph:
    """Automate -> graphe oriente etiquete (pour l'isomorphisme)."""
    G = nx.MultiDiGraph()
    for s in aut.states:
        G.add_node(s, initial=(s == aut.initial), accepting=(s in aut.accepting))
    for src, sym, dst in aut.transitions:
        G.add_edge(src, dst, symbol=sym)
    return G


def is_isomorphic(a: Automaton, b: Automaton) -> bool:
    """Isomorphisme structurel, independant du nommage des etats."""
    nm = isomorphism.categorical_node_match(["initial", "accepting"], [False, False])
    em = isomorphism.categorical_multiedge_match("symbol", None)
    matcher = isomorphism.MultiDiGraphMatcher(
        to_graph(a), to_graph(b), node_match=nm, edge_match=em
    )
    return matcher.is_isomorphic()



def compare(a: Automaton, b: Automaton) -> dict:
    scores = {
        "states":      jaccard(a.states, b.states),
        "alphabet":    jaccard(a.alphabet, b.alphabet),
        "accepting":   jaccard(a.accepting, b.accepting),   # brut (garde pour debug)
        "initial":     1.0 if a.initial == b.initial else 0.0,  # brut
        "transitions": jaccard(a.transitions, b.transitions),   # brut
    }

    report = {
        "scores": scores,
        "isomorphic": is_isomorphic(a, b),
        "missing_states": a.states - b.states,
        "extra_states": b.states - a.states,
        "missing_transitions": a.transitions - b.transitions,
        "extra_transitions": b.transitions - a.transitions,
    }

    mapping, matched = find_best_mapping(b, a)
    total = max(len(a.transitions), len(b.transitions))

    report["mapping"] = mapping
    report["aligned_transitions"] = matched / total if total else 1.0
    report["aligned_initial"] = aligned_initial_score(mapping, b, a)
    report["aligned_accepting"] = aligned_accepting_score(mapping, b, a)

    aligned_parts = [
        scores["states"],
        scores["alphabet"],
        report["aligned_accepting"],
        report["aligned_initial"],
        report["aligned_transitions"],
    ]
    report["global_aligned"] = sum(aligned_parts) / len(aligned_parts)

    # garde aussi le global brut pour comparaison
    report["global"] = sum(scores.values()) / len(scores)

    return report








def print_report(result: dict) -> None:
    s = result["scores"]
    print(f"  etats       : {s['states']:.1%}")
    print(f"  alphabet    : {s['alphabet']:.1%}")
    print(f"  acceptants  : {s['accepting']:.1%}")
    print(f"  initial     : {s['initial']:.1%}")
    print(f"  transitions : {s['transitions']:.1%}")
    print(f"  GLOBAL      : {result['global']:.1%}  "
          f"(difference {(1 - result['global']):.1%})")
    print(f"  isomorphe   : {'oui' if result['isomorphic'] else 'non'}")
    if result["missing_transitions"]:
        print(f"  manquantes  : {len(result['missing_transitions'])}")
        for t in sorted(result["missing_transitions"]):
            print(f"    - {t[0]}:{t[1]}>{t[2]}")
    if result["extra_transitions"]:
        print(f"  en trop     : {len(result['extra_transitions'])}")
        for t in sorted(result["extra_transitions"]):
            print(f"    + {t[0]}:{t[1]}>{t[2]}")


# ---------- test ----------

if __name__ == "__main__":
    automate_A = """#states
s0
s1
s2
s3
#initial
s0
#accepting
s1
s3
#alphabet
a
b
c
#transitions
s0:a>s0
s0:b>s2
s0:c>s2
s1:a>s0
s1:b>s3
s1:c>s1
s2:a>s1
s2:b>s2
s2:c>s2
s3:a>s2
s3:b>s0
s3:c>s2
"""

    automate_B = """#states
s0
s1
s2
s3
#initial
s0
#accepting
s1
s2
#alphabet
a
b
c
#transitions
s0:a>s0
s0:b>s1
s0:c>s0
s1:a>s0
s1:b>s1
s1:c>s3
s2:a>s0
s2:b>s2
s2:c>s2
s3:a>s2
s3:b>s1
s3:c>s0
"""

    a = parse_table(automate_A, is_path=False)
    b = parse_table(automate_B, is_path=False)

    result = compare(a, b)
    print_report(result)
