"""
Serialisation du modele kNN dans un format binaire brut, portable Python<->C++.

Layout du fichier (little-endian) :
    [int32]            N        nombre d'exemples
    [int32]            D        dimension des vecteurs de features
    [int32]            C        nombre de classes (labels distincts)
    [N * D * float32]  X        vecteurs de features, ligne par ligne:wq
    [N * int32]        y        label de chaque exemple (index de classe 0..C-1)
    [C blocs]          labels   pour chaque classe : [int32 len][len bytes utf-8]

Cote C++ : lire N, D, C ; puis X (N*D floats) ; puis y (N ints) ;
puis pour chaque classe lire la longueur puis les octets du caractere.
"""

import numpy as np


def save_model(path, X, y):
    """
    Sauve (X, y) au format binaire portable.

    Args:
        path: chemin de sortie (.bin par convention)
        X: array (N, D) float
        y: array (N,) de labels (str ou char)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    N, D = X.shape

    classes = sorted(set(y.tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(classes)}
    y_idx = np.array([label_to_idx[lab] for lab in y], dtype=np.int32)
    C = len(classes)

    with open(path, "wb") as f:
        np.array([N, D, C], dtype=np.int32).tofile(f)
        X.tofile(f)                      # N*D float32
        y_idx.tofile(f)                  # N int32
        for lab in classes:
            b = str(lab).encode("utf-8")
            np.array([len(b)], dtype=np.int32).tofile(f)
            f.write(b)


def load_model(path):
    """
    Recharge (X, y) depuis le format binaire.

    Returns:
        X (N, D) float32, y (N,) array de labels (str)
    """
    with open(path, "rb") as f:
        N, D, C = np.fromfile(f, dtype=np.int32, count=3)
        X = np.fromfile(f, dtype=np.float32, count=N * D).reshape(N, D)
        y_idx = np.fromfile(f, dtype=np.int32, count=N)
        classes = []
        for _ in range(C):
            length = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            label = f.read(length).decode("utf-8")
            classes.append(label)
    y = np.array([classes[i] for i in y_idx])
    return X, y


# ---------- test round-trip ----------
if __name__ == "__main__":
    X = np.random.rand(50, 144).astype(np.float32)
    y = np.array(list("abcde") * 10)

    save_model("test_model.bin", X, y)
    X2, y2 = load_model("test_model.bin")

    assert np.allclose(X, X2), "X ne correspond pas"
    assert (y == y2).all(), "y ne correspond pas"
    print(f"round-trip OK : {X2.shape[0]} exemples, dim {X2.shape[1]}, "
          f"classes {sorted(set(y2.tolist()))}")

    import os
    print(f"taille fichier : {os.path.getsize('test_model.bin')} octets")
