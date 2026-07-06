import subprocess
from pathlib import Path
from typing import Optional

from automaton_parser import Automaton, parse_table


def to_dot(aut: Automaton, *, rankdir: str = "LR") -> str:
    """
    Convertit un Automaton en code DOT (format Graphviz).
    
    Args:
        aut: l'automate a rendre.
        rankdir: orientation du graphe ("LR" gauche-droite, "TB" haut-bas).
    
    Returns:
        Code DOT sous forme de string.
    """
    lines = ["digraph automaton {"]
    lines.append(f"  rankdir={rankdir};")
    lines.append('  node [shape=circle, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica"];')
    
    if aut.initial is not None:
        lines.append('  __start__ [shape=point, width=0.1, label=""];')
    
    for state in sorted(aut.states):
        if state in aut.accepting:
            lines.append(f'  "{state}" [shape=doublecircle];')
        else:
            lines.append(f'  "{state}";')
    
    if aut.initial is not None:
        lines.append(f'  __start__ -> "{aut.initial}";')
    grouped: dict[tuple[str, str], list[str]] = {}
    for src, sym, dst in aut.transitions:
        grouped.setdefault((src, dst), []).append(sym)
    
    for (src, dst), syms in sorted(grouped.items()):
        label = ", ".join(sorted(syms))
        lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')
    
    lines.append("}")
    return "\n".join(lines)


def render(
    aut: Automaton,
    output_path: str,
    *,
    fmt: str = "png",
    engine: str = "dot",
    rankdir: str = "LR",
    keep_dot: bool = False,
) -> Path:
    """
    Rend un automate en image via Graphviz.
    
    Args:
        aut: l'automate a rendre.
        output_path: chemin de sortie (sans extension, elle sera ajoutee).
        fmt: format de sortie ("png", "svg", "pdf"...).
        engine: moteur Graphviz ("dot", "neato", "circo", "fdp", "twopi").
                "dot" : layout hierarchique, le plus lisible pour les automates.
                "circo" : disposition circulaire, joli pour les petits automates.
                "neato"/"fdp" : layout par forces, moins structure.
        rankdir: orientation ("LR", "TB", "RL", "BT").
        keep_dot: si True, garde le fichier .dot intermediaire.
    
    Returns:
        Path vers le fichier image genere.
    """
    output = Path(output_path).with_suffix(f".{fmt}")
    dot_path = output.with_suffix(".dot")
    
    dot_code = to_dot(aut, rankdir=rankdir)
    dot_path.write_text(dot_code, encoding="utf-8")
    
    try:
        subprocess.run(
            [engine, f"-T{fmt}", str(dot_path), "-o", str(output)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Graphviz '{engine}' introuvable. Installe avec: "
            "sudo apt install graphviz   (ou: brew install graphviz)"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Echec Graphviz: {e.stderr.decode()}")
    
    if not keep_dot:
        dot_path.unlink()
    
    return output


# ---------- test sur les deux automates fournis ----------

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
    
    print("=== DOT genere pour l'automate A ===")
    print(to_dot(a))
    print()
    
    path_a = render(a, ".", fmt="png")
    path_b = render(b, ".", fmt="png")
    
    print(f"Image A generee : {path_a}")
    print(f"Image B generee : {path_b}")
