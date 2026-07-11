"""
Separation des tetes de fleches fusionnees.

Port fidele de la version C++ (cpp/src/split_merged_tips.cc) : meme logique,
memes seuils, meme cv2.kmeans, pour validation croisee Python <-> C++.

Un blob dont l'aire depasse merge_area_ratio * mediane des aires est
suspect ; on tente de le scinder en deux sous-pointes par k-means (k=2) sur
les positions des pixels. La scission n'est retenue que si les deux moities
sont plausibles (taille suffisante, centres nettement separes) ; sinon le
blob est laisse inchange.

Format des tips (identique au reste du pipeline Python) :
    {"centroid": (cx, cy), "area": int, "bbox": (x, y, w, h),
     "pixels": (ys, xs)}   # pixels au format np.where

Note validation croisee : cv2.kmeans utilise un RNG interne. Pour des
resultats reproductibles entre executions et entre langages, fixer la graine
avec cv2.setRNGSeed(0) des deux cotes AVANT l'appel.
"""

import cv2
import numpy as np


def _make_tip(ys, xs):
    """Construit un tip a partir de coordonnees de pixels (arrays ys, xs)."""
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    return {
        "centroid": (float(xs.mean()), float(ys.mean())),
        "area": int(len(xs)),
        "bbox": (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
        "pixels": (ys, xs),
    }


def _try_split_tip(tip, median_area):
    """
    Tente de scinder un blob en 2 sous-pointes via k-means sur les positions.
    Retourne (tip0, tip1) ou None si la scission n'est pas plausible.
    """
    ys, xs = tip["pixels"]
    if len(xs) < 8:
        return None

    # points en (x, y), float32, comme le C++
    points = np.column_stack([xs, ys]).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        points, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    mask0 = labels == 0
    part0_x, part0_y = xs[mask0], ys[mask0]
    part1_x, part1_y = xs[~mask0], ys[~mask0]

    # plausibilite : chaque moitie doit ressembler a une pointe normale
    min_part = 0.35 * median_area
    if len(part0_x) < min_part or len(part1_x) < min_part:
        return None

    # centres nettement separes (sinon blob compact, pas deux pointes)
    dx = centers[0, 0] - centers[1, 0]
    dy = centers[0, 1] - centers[1, 1]
    center_dist = float(np.sqrt(dx * dx + dy * dy))
    typical_size = float(np.sqrt(median_area))
    if center_dist < 0.8 * typical_size:
        return None

    return _make_tip(part0_y, part0_x), _make_tip(part1_y, part1_x)


def split_merged_tips(tips, merge_area_ratio=1.6):
    """
    Scinde les tetes de fleches fusionnees. Modifie et retourne la liste tips.
    A appeler en fin de find_triangles, avant le return.
    """
    if len(tips) < 2:
        return tips  # pas de mediane fiable

    median_area = float(np.median([t["area"] for t in tips]))

    result = []
    for tip in tips:
        if tip["area"] <= merge_area_ratio * median_area:
            result.append(tip)
            continue
        split = _try_split_tip(tip, median_area)
        if split is not None:
            result.append(split[0])
            result.append(split[1])
        else:
            result.append(tip)  # repli : blob inchange
    return result


# ---------- test ----------
if __name__ == "__main__":
    cv2.setRNGSeed(0)  # reproductibilite

    def make_fake_tip(cx, cy, n):
        xs = np.array([cx + i % 12 for i in range(n)])
        ys = np.array([cy + i // 12 for i in range(n)])
        return _make_tip(ys, xs)

    tips = [
        make_fake_tip(10, 10, 120),
        make_fake_tip(50, 10, 120),
        make_fake_tip(90, 10, 120),
    ]
    # blob fusionne : deux amas distants (aire 2x la normale)
    xs = np.array([130 + i % 10 for i in range(120)] + [160 + i % 10 for i in range(120)])
    ys = np.array([10 + i // 10 for i in range(120)] + [10 + i // 10 for i in range(120)])
    tips.append(_make_tip(ys, xs))

    print("avant:", len(tips), "aires:", [t["area"] for t in tips])
    tips = split_merged_tips(tips)
    print("apres:", len(tips), "aires:", [t["area"] for t in tips])
    assert len(tips) == 5, "le blob fusionne aurait du etre scinde"
    print("OK : blob fusionne -> 2 pointes")
