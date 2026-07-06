import argparse
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
)
from model_io import load_model
from classifier import KNNClassifier
from features import compute_hog

DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "data" / "knn_model.bin")


def segment_automaton(image_path, classifier=None, debug=False):
    """
    Execute toute la chaine de segmentation sur une image d'automate.

    Returns:
        dict {
            "states": liste d'etats {center_x, center_y, radius, outer_radius,
                                     accepting, ...},
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
                features = compute_hog(label["crop"])
                label["symbol"] = classifier.predict(features)

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

    print(f"{len(states)} etat(s) :")
    for i, s in enumerate(states):
        flags = []
        if i == initial:
            flags.append("initial")
        if s.get("accepting"):
            flags.append("acceptant")
        tag = f" [{', '.join(flags)}]" if flags else ""
        print(f"  s{i}: centre=({s['center_x']:.0f}, {s['center_y']:.0f}){tag}")

    print(f"\n{len(arrows)} transition(s) detectee(s) :")
    for arrow in arrows:
        # symboles reconnus sur cette arete (ignore les None non reconnus)
        symbols = [lbl.get("symbol") for lbl in arrow.get("labels", [])]
        symbols = [s for s in symbols if s is not None]
        if symbols:
            sym_str = ", ".join(str(s) for s in symbols)
            print(f"  s{arrow['source']} --{sym_str}--> s{arrow['dest']}")
        else:
            # pas de reconnaissance (pas de classifieur, ou symbole rejete)
            n_labels = len(arrow.get("labels", []))
            print(f"  s{arrow['source']} -> s{arrow['dest']}  "
                  f"({n_labels} etiquette(s) non reconnue(s))")

    print("\nMatrice d'adjacence :")
    n = len(states)
    print("     " + "  ".join(f"s{j}" for j in range(n)))
    for i, row in enumerate(result["matrix"]):
        cells = "   ".join("1" if cell else "0" for cell in row)
        print(f"s{i}:  {cells}")


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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="modele kNN entraine (defaut: data/knn_model.bin)")
    parser.add_argument("--debug", action="store_true",
                        help="conserve et affiche les images intermediaires")
    args = parser.parse_args()

    classifier = None
    try:
        classifier = load_classifier(args.model)
    except FileNotFoundError:
        print(f"[!] modele {args.model} introuvable : "
              f"segmentation seule, sans reconnaissance des symboles.")

    result = segment_automaton(args.image, classifier=classifier,
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
