
from pathlib import Path
from random import randint

from automaton_parser import Automaton, to_table
from automaton_render import render

ALPHABET_POOL = "vbcdnz"


def _pick(seq):
    return seq[randint(0, len(seq) - 1)]


def _sample(pool: str, k: int) -> list[str]:
    """Tire k lettres distinctes du pool, sans remise (sans random.sample)."""
    available = list(pool)
    picked = []
    for _ in range(k):
        idx = randint(0, len(available) - 1)
        picked.append(available.pop(idx))
    return picked


def generate_dfa(min_states: int, max_states: int,
                 min_alpha: int, max_alpha: int) -> Automaton:
    """DFA aleatoire, complet (transition definie pour chaque (q, a))."""
    n = randint(min_states, max_states)
    k = randint(min_alpha, max_alpha)

    states = [f"s{i}" for i in range(n)]
    alphabet = _sample(ALPHABET_POOL, k)  # lettres aleatoires, pas un prefixe

    n_accept = randint(1, max(1, n - 1))
    candidates = states.copy()
    accepting = set()
    for _ in range(n_accept):
        idx = randint(0, len(candidates) - 1)
        accepting.add(candidates.pop(idx))

    transitions = set()
    for q in states:
        for a in alphabet:
            transitions.add((q, a, _pick(states)))

    return Automaton(
        states=set(states),
        initial="s0",
        accepting=accepting,
        alphabet=set(alphabet),
        transitions=transitions,
    )


# parametres par niveau (etats min/max, alphabet min/max)
LEVELS = {
    "simple_dfa": (2, 3, 2, 2),  # 2-3 etats, 2 lettres
    "medium_dfa": (3, 4, 3, 3),  # 3-4 etats, 3 lettres
    "hard_dfa":   (5, 6, 4, 5),  # 5-6 etats, 4-5 lettres
}


def generate_corpus(out_dir: str, n_per_level: int = 50,
                    render_images: bool = True) -> dict:
    """
    Genere n_per_level DFA par niveau, sauve les tables en .txt
    et optionnellement les images en .png.
    Arborescence : out_dir/{simple_dfa,medium_dfa,hard_dfa}/dfa_NNN.{txt,png}
    """
    base = Path(out_dir)
    stats = {}

    for level, params in LEVELS.items():
        level_dir = base / level
        level_dir.mkdir(parents=True, exist_ok=True)
        sizes = []

        for i in range(n_per_level):
            aut = generate_dfa(*params)
            stem = level_dir / f"dfa_{i:03d}"

            (stem.with_suffix(".txt")).write_text(to_table(aut), encoding="utf-8")
            if render_images:
                render(aut, str(stem), fmt="png")

            sizes.append((len(aut.states), len(aut.alphabet)))

        stats[level] = {
            "count": n_per_level,
            "states_range": (min(s[0] for s in sizes), max(s[0] for s in sizes)),
            "alpha_range":  (min(s[1] for s in sizes), max(s[1] for s in sizes)),
        }

    return stats


# ---------- test ----------

if __name__ == "__main__":
    stats = generate_corpus(".", n_per_level=50)
    for level, info in stats.items():
        print(f"{level:7s} | {info['count']} DFA | "
              f"etats {info['states_range']} | "
              f"alphabet {info['alpha_range']}")
