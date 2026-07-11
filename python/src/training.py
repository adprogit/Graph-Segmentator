import subprocess
from pathlib import Path

import cv2
import numpy as np

from model_io import save_model
from classifier import KNNClassifier
from features import normalize_crop, compute_hog

# alphabets des deux modeles : lettres pour les symboles de transitions et
# le 's' des noms d'etats, chiffres pour la suite des noms d'etats
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")


# =============================================================================
# 1. RENDU GRAPHVIZ D'UN CARACTERE
# =============================================================================

def render_char_graphviz(char, fontsize=24, fontname="Helvetica"):
    """
    Rend un caractere seul via Graphviz (shape=none => glyphe sans bordure).
    Retourne une image niveaux de gris (fond blanc, trait noir) ou None.

    On passe le DOT sur stdin et on recupere le PNG sur stdout : pas de
    fichier temporaire.
    """
    safe = char.replace("\\", "\\\\").replace('"', '\\"')
    dot = (
        f'digraph {{ node [shape=none, fontname="{fontname}", '
        f'fontsize={fontsize}]; n [label="{safe}"]; }}'
    )
    result = subprocess.run(
        ["dot", "-Tpng"], input=dot.encode(), capture_output=True
    )
    if result.returncode != 0 or not result.stdout:
        return None
    buf = np.frombuffer(result.stdout, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


# =============================================================================
# 2. NORMALISATION  (DOIT etre identique cote inference)
# =============================================================================

def to_binary_trait(gray):
    """Binarise : trait=255, fond=0 (meme convention que le pipeline vision)."""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# =============================================================================
# 3. AUGMENTATION
# =============================================================================

def augment(binary, rng):
    """
    Applique des perturbations aleatoires a un glyphe binaire pour simuler
    la variabilite des crops reels (epaisseur, inclinaison, bruit, decalage).
    Entree/sortie : binaire (trait=255, fond=0).
    """
    img = binary.copy()

    thickness = rng.choice([-1, 0, 0, 1])  # biais vers "inchange"
    if thickness == 1:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    elif thickness == -1:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))

    angle = rng.uniform(-8, 8)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

    tx = rng.integers(-2, 3)
    ty = rng.integers(-2, 3)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

    if rng.random() < 0.5:
        noise = rng.random(img.shape)
        img[noise < 0.02] = 255
        img[noise > 0.98] = 0
        img = cv2.medianBlur(img, 3)

    return img


def build_char_base(char, fontname="Helvetica"):
    """Rend un caractere une seule fois par fontsize -> liste d'images binaires.

    14 est la taille reelle des etiquettes Graphviz (defaut) : indispensable
    pour que le kNN a 26 classes generalise aux vraies images.
    """
    bases = []
    for fontsize in (14, 18, 24, 30):
        gray = render_char_graphviz(char, fontsize=fontsize, fontname=fontname)
        if gray is None:
            continue
        bases.append(to_binary_trait(gray))
    return bases


def generate_dataset(alphabet, n_per_char=2000, fontname="Helvetica", seed=0):
    """
    Genere (X, y) : pour chaque caractere, rend quelques bases (par fontsize)
    puis produit n_per_char variantes augmentees + normalisees + HOG.

    Les vecteurs partages par plusieurs classes sont retires : l'erosion
    peut reduire un glyphe fin (taille 14) a un residu degenere dont le HOG
    est identique d'une classe a l'autre (ambigu, et source d'egalites
    parfaites dont l'issue depend des arrondis, donc non portables).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for char in alphabet:
        bases = build_char_base(char, fontname=fontname)
        if not bases:
            print(f"  [!] rendu echoue pour '{char}', ignore")
            continue
        for _ in range(n_per_char):
            # l'erosion peut effacer completement le glyphe : on retire
            while True:
                base = bases[rng.integers(len(bases))]
                aug = augment(base, rng)
                if aug.any():
                    break
            norm = normalize_crop(aug)
            X.append(compute_hog(norm))
            y.append(char)

    # deduplication inter-classes (cf. docstring)
    owners = {}
    for features, char in zip(X, y):
        owners.setdefault(features.tobytes(), set()).add(char)
    kept = [i for i, features in enumerate(X)
            if len(owners[features.tobytes()]) == 1]
    if len(kept) < len(X):
        print(f"  {len(X) - len(kept)} exemples degeneres retires "
              f"(partages entre classes)")
    X = [X[i] for i in kept]
    y = [y[i] for i in kept]

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def confusion_report(y_true, y_pred, labels):
    """Matrice de confusion + accuracy, sans sklearn."""
    label_to_i = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    correct = 0
    for t, p in zip(y_true, y_pred):
        cm[label_to_i[t], label_to_i[p]] += 1
        if t == p:
            correct += 1
    accuracy = correct / len(y_true)
    return cm, accuracy


# =============================================================================
# ENTRAINEMENT COMPLET D'UN MODELE
# =============================================================================

def train_and_save(alphabet, model_path, n_per_char=400, seed=0):
    """
    Genere le dataset d'un alphabet, entraine et evalue un kNN, puis
    sauvegarde le modele (tous les exemples) dans model_path.
    """
    print("Generation du dataset...")
    X, y = generate_dataset(alphabet, n_per_char=n_per_char, seed=seed)
    print(f"  {len(X)} exemples, dim HOG = {X.shape[1]}")

    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.2, seed=seed)
    print(f"  train={len(X_tr)}  test={len(X_te)}")

    print("Entrainement kNN...")
    clf = KNNClassifier(k=3, weighted=True)
    clf.fit(X_tr, y_tr)

    print("Evaluation...")
    y_pred = clf.predict(X_te)
    cm, acc = confusion_report(y_te, y_pred, alphabet)

    print(f"\nAccuracy : {acc:.1%}\n")
    print("Matrice de confusion (lignes=vrai, colonnes=predit) :")
    print("      " + "  ".join(f"{c:>3}" for c in alphabet))
    for i, c in enumerate(alphabet):
        row = "  ".join(f"{cm[i, j]:>3}" for j in range(len(alphabet)))
        print(f"  {c} : {row}")

    save_model(model_path, X, y)
    print(f"\nModele sauvegarde dans {model_path} "
          f"({len(X)} exemples, dim {X.shape[1]}).")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    DATA = Path(__file__).resolve().parents[2] / "data"
    print("--- Modele lettres (a-z) ---")
    train_and_save(LETTERS, str(DATA / "knn_letters.bin"))
    print("\n--- Modele chiffres (0-9) ---")
    train_and_save(DIGITS, str(DATA / "knn_digits.bin"))
