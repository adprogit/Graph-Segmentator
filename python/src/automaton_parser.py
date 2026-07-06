
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Automaton:
    states: set[str] = field(default_factory=set)
    initial: Optional[str] = None
    accepting: set[str] = field(default_factory=set)
    alphabet: set[str] = field(default_factory=set)
    # transitions stockees comme ensemble de triplets (src, sym, dst)
    # -> set d'ensembles pour comparaison directe via & | -
    transitions: set[tuple[str, str, str]] = field(default_factory=set)

    def is_deterministic(self) -> bool:
        """DFA ssi pas d'epsilon et au plus une transition par (etat, symbole)."""
        if any(sym == "" or sym.lower() in ("eps", "epsilon", "ε") 
               for _, sym, _ in self.transitions):
            return False
        seen = set()
        for src, sym, _ in self.transitions:
            if (src, sym) in seen:
                return False
            seen.add((src, sym))
        return True

    def has_epsilon(self) -> bool:
        return any(sym == "" or sym.lower() in ("eps", "epsilon", "ε")
                   for _, sym, _ in self.transitions)

    def automaton_type(self) -> str:
        if self.has_epsilon():
            return "epsilon-NFA"
        if self.is_deterministic():
            return "DFA"
        return "NFA"

    def __repr__(self) -> str:
        return (f"Automaton(type={self.automaton_type()}, "
                f"|Q|={len(self.states)}, |Σ|={len(self.alphabet)}, "
                f"|δ|={len(self.transitions)}, q0={self.initial}, "
                f"F={self.accepting})")


def parse_table(source: str, *, is_path: bool = True) -> Automaton:
    """
    Parse une table d'automate.
    
    Args:
        source: chemin de fichier ou contenu texte directement.
        is_path: True si source est un chemin, False si c'est le texte.
    
    Returns:
        Un objet Automaton pret a etre compare / converti en graphe.
    """
    if is_path:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = source

    aut = Automaton()
    current_section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("#"):
            current_section = line[1:].strip().lower()
            continue

        if current_section == "states":
            aut.states.add(line)
        elif current_section == "initial":
            if aut.initial is not None:
                raise ValueError(f"Plusieurs etats initiaux declares: {aut.initial}, {line}")
            aut.initial = line
        elif current_section == "accepting":
            aut.accepting.add(line)
        elif current_section == "alphabet":
            aut.alphabet.add(line)
        elif current_section == "transitions":
            # format attendu: src:sym>dst
            try:
                src, rest = line.split(":", 1)
                sym, dst = rest.split(">", 1)
            except ValueError:
                raise ValueError(f"Transition mal formee: {line!r}")
            aut.transitions.add((src.strip(), sym.strip(), dst.strip()))
        else:
            raise ValueError(f"Ligne hors section: {line!r}")

    _validate(aut)
    return aut


def to_table(aut: Automaton) -> str:
    """Serialise un Automaton au format table sectionne (inverse de parse_table)."""
    parts = ["#states"]
    parts.extend(sorted(aut.states))
    parts.append("#initial")
    parts.append(aut.initial if aut.initial is not None else "")
    parts.append("#accepting")
    parts.extend(sorted(aut.accepting))
    parts.append("#alphabet")
    parts.extend(sorted(aut.alphabet))
    parts.append("#transitions")
    for src, sym, dst in sorted(aut.transitions):
        parts.append(f"{src}:{sym}>{dst}")
    return "\n".join(parts) + "\n"


def _validate(aut: Automaton) -> None:
    """Verifie la coherence interne. Leve ValueError si incoherent."""
    if aut.initial is not None and aut.initial not in aut.states:
        raise ValueError(f"Etat initial {aut.initial!r} absent de #states")
    
    unknown_accepting = aut.accepting - aut.states
    if unknown_accepting:
        raise ValueError(f"Etats acceptants inconnus: {unknown_accepting}")
    
    for src, sym, dst in aut.transitions:
        if src not in aut.states:
            raise ValueError(f"Transition depuis etat inconnu: {src}")
        if dst not in aut.states:
            raise ValueError(f"Transition vers etat inconnu: {dst}")
        # symbole epsilon autorise meme s'il n'est pas dans l'alphabet
        is_eps = sym == "" or sym.lower() in ("eps", "epsilon", "ε")
        if not is_eps and sym not in aut.alphabet:
            raise ValueError(f"Symbole {sym!r} absent de #alphabet")


# ---------- test direct ----------

if __name__ == "__main__":
    exemple = """#states
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
    aut = parse_table(exemple, is_path=False)
    print(aut)
    print(f"  etats           : {sorted(aut.states)}")
    print(f"  initial         : {aut.initial}")
    print(f"  acceptants      : {sorted(aut.accepting)}")
    print(f"  alphabet        : {sorted(aut.alphabet)}")
    print(f"  nb transitions  : {len(aut.transitions)}")
    print(f"  type detecte    : {aut.automaton_type()}")
