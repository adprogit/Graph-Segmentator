import argparse
import re
from pathlib import Path


from segmentation import (
    load_image,
    to_binary,
    segment_states,
    remove_states_from_img,
    merge_states_and_acceptants,
    find_triangles,
    build_adjacency_matrix,
    isolate_labels,
    assign_labels_to_arrows,
    segment_name_characters,
)
from model_io import load_model
from classifier import KNNClassifier
from features import compute_hog, normalize_crop

_DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_LETTERS = str(_DATA / "knn_letters.bin")
DEFAULT_DIGITS = str(_DATA / "knn_digits.bin")

_NAME_PATTERN = re.compile(r"s[0-9]+")


def recognize_state_name(name_crop, letters_classifier, digits_classifier):
    """
    Lit le nom d'un etat depuis son crop : premier caractere via le modele
    lettres (le 's'), les suivants via le modele chiffres.
    Retourne la chaine lue, ou None si le crop n'a pas au moins 2 caracteres.
    """
    crops = segment_name_characters(name_crop)
    if len(crops) < 2:
        return None
    # normalize_crop recadre au ras du trait, comme a l'entrainement
    chars = [letters_classifier.predict(compute_hog(normalize_crop(crops[0])))]
    for crop in crops[1:]:
        chars.append(digits_classifier.predict(compute_hog(normalize_crop(crop))))
    if any(c is None for c in chars):
        return None
    return "".join(chars)


def assign_state_names(states, raw_names):
    """
    Attribue son nom final a chaque etat (champ state["name"]).

    Un nom brut est retenu s'il est valide (motif s[0-9]+) et unique parmi
    les noms bruts ; sinon l'etat retombe sur le plus petit nom d'indice
    s{i} encore libre, dans l'ordre des etats. Ordre fige : le portage C++
    doit le reproduire a l'identique.
    """
    valid = [name if name is not None and _NAME_PATTERN.fullmatch(name)
             else None
             for name in raw_names]
    counts = {}
    for name in valid:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1

    used = {name for name in valid if name is not None and counts[name] == 1}
    for state, name in zip(states, valid):
        state["name"] = name if name is not None and counts[name] == 1 else None

    next_index = 0
    for state in states:
        if state["name"] is not None:
            continue
        while f"s{next_index}" in used:
            next_index += 1
        state["name"] = f"s{next_index}"
        used.add(state["name"])
    return states


def segment_automaton(image_path, classifier=None, digit_classifier=None,
                      debug=False):
    """
    Execute toute la chaine de segmentation sur une image d'automate.

    classifier reconnait les lettres (symboles de transitions et premier
    caractere des noms d'etats), digit_classifier les chiffres (suite des
    noms). Les noms ne sont lus que si les deux modeles sont fournis ;
    sinon chaque etat garde son nom d'indice s{i}.

    Returns:
        dict {
            "states": liste d'etats {center_x, center_y, radius, outer_radius,
                                     accepting, name, ...},
            "matrix": matrice d'adjacence (matrix[src][dst] = arete | 0),
            "arrows": liste d'aretes {source, dest, chemin, labels, ...},
            "initial": index de l'etat initial (ou None),
            "images": images intermediaires si debug=True,
        }
    """
    # 1. chargement + binarisation (trait=255, fond=0)
    gray = load_image(image_path)
    if gray is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    binary = to_binary(gray)

    states = segment_states(binary)

    img_without_inner = remove_states_from_img(binary, states)
    residual_circles = segment_states(img_without_inner)
    all_states = merge_states_and_acceptants(states, residual_circles)
    img_clean = remove_states_from_img(binary, all_states)
    tips = find_triangles(img_clean)
    n_states = len(all_states)
    matrix = [[0] * n_states for _ in range(n_states)]
    matrix, initial_index, arrows = build_adjacency_matrix(
        matrix, all_states, tips, img_clean
    )

    img_labels = isolate_labels(img_clean, tips, arrows)

    assign_labels_to_arrows(img_labels, arrows)

    if classifier is not None:
        for edge in arrows:
            for label in edge["labels"]:
                # normalize_crop recadre au ras du trait : meme cadrage
                # que les glyphes d'entrainement (crop_to_content)
                features = compute_hog(normalize_crop(label["crop"]))
                label["symbol"] = classifier.predict(features)

    if classifier is not None and digit_classifier is not None:
        raw_names = [recognize_state_name(s["name_crop"], classifier,
                                          digit_classifier)
                     for s in all_states]
    else:
        raw_names = [None] * len(all_states)
    assign_state_names(all_states, raw_names)

    result = {
        "states": all_states,
        "matrix": matrix,
        "arrows": arrows,
        "initial": initial_index,
    }
    if debug:
        result["images"] = {
            "binary": binary,
            "img_clean": img_clean,
            "img_labels": img_labels,
        }
    return result


def print_summary(result):
    """Affiche un resume textuel de la segmentation."""
    states = result["states"]
    arrows = result["arrows"]
    initial = result["initial"]
    names = [s.get("name") or f"s{i}" for i, s in enumerate(states)]

    print(f"{len(states)} etat(s) :")
    for i, s in enumerate(states):
        flags = []
        if i == initial:
            flags.append("initial")
        if s.get("accepting"):
            flags.append("acceptant")
        tag = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {names[i]}: centre=({s['center_x']:.0f}, {s['center_y']:.0f}){tag}")

    print(f"\n{len(arrows)} transition(s) detectee(s) :")
    for arrow in arrows:
        # symboles reconnus sur cette arete (ignore les None non reconnus)
        symbols = [lbl.get("symbol") for lbl in arrow.get("labels", [])]
        symbols = [s for s in symbols if s is not None]
        src, dst = names[arrow["source"]], names[arrow["dest"]]
        if symbols:
            sym_str = ", ".join(str(s) for s in symbols)
            print(f"  {src} --{sym_str}--> {dst}")
        else:
            # pas de reconnaissance (pas de classifieur, ou symbole rejete)
            n_labels = len(arrow.get("labels", []))
            print(f"  {src} -> {dst}  "
                  f"({n_labels} etiquette(s) non reconnue(s))")

    print("\nMatrice d'adjacence :")
    width = max(len(n) for n in names) if names else 2
    print(" " * (width + 3) + "  ".join(f"{n:>{width}}" for n in names))
    for i, row in enumerate(result["matrix"]):
        cells = "   ".join("1" if cell else "0" for cell in row)
        print(f"{names[i]:>{width}}:  {cells}")


def load_classifier(model_path, k=3):
    """Charge le modele kNN depuis un fichier binaire et le prepare."""
    X_train, y_train = load_model(model_path)
    clf = KNNClassifier(k=k, weighted=True)
    clf.fit(X_train, y_train)      # fit = stocker les vecteurs (instantane)
    return clf


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation + reconnaissance d'une image d'automate."
    )
    parser.add_argument("image", help="chemin vers l'image de l'automate")
    parser.add_argument("--letters", default=DEFAULT_LETTERS,
                        help="modele kNN lettres (defaut: data/knn_letters.bin)")
    parser.add_argument("--digits", default=DEFAULT_DIGITS,
                        help="modele kNN chiffres (defaut: data/knn_digits.bin)")
    parser.add_argument("--debug", action="store_true",
                        help="conserve et affiche les images intermediaires")
    args = parser.parse_args()

    classifier = None
    digit_classifier = None
    try:
        classifier = load_classifier(args.letters)
    except FileNotFoundError:
        print(f"[!] modele {args.letters} introuvable : "
              f"segmentation seule, sans reconnaissance des symboles.")
    try:
        digit_classifier = load_classifier(args.digits)
    except FileNotFoundError:
        print(f"[!] modele {args.digits} introuvable : "
              f"noms d'etats non reconnus.")

    result = segment_automaton(args.image, classifier=classifier,
                               digit_classifier=digit_classifier,
                               debug=args.debug)
    print_summary(result)

    if args.debug:
        import matplotlib.pyplot as plt
        imgs = result["images"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (name, img) in zip(axes, imgs.items()):
            ax.imshow(img, cmap="gray")
            ax.set_title(name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
