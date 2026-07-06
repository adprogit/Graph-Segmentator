"""
Features partagees entre l'ENTRAINEMENT et l'INFERENCE.

Source de verite unique : train_pipeline.py et main.py importent ces fonctions.
Si ces implementations divergent entre train et inference, la reconnaissance
s'effondre (vecteurs incomparables). Ne pas dupliquer ailleurs.

Portable C++ ligne a ligne (numpy + cv2 -> Eigen/OpenCV C++).
"""

import cv2
import numpy as np


def crop_to_content(binary):
    """Recadre sur la bounding box du trait."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return binary
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalize_crop(binary, target_size=32):
    """Centre le glyphe dans un carre (preserve ratio) puis redimensionne."""
    content = crop_to_content(binary)
    h, w = content.shape
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    yo, xo = (side - h) // 2, (side - w) // 2
    square[yo:yo + h, xo:xo + w] = content
    return cv2.resize(square, (target_size, target_size),
                      interpolation=cv2.INTER_NEAREST)


def compute_hog(img, cell_size=8, n_bins=9):
    """
    HOG maison. img de taille fixe (ex 32x32).
    Gradients Sobel sur l'image entiere, orientation non signee [0, pi),
    histogramme pondere par la magnitude par cellule, normalisation L2.
    """
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) % np.pi

    h, w = img.shape
    bin_width = np.pi / n_bins
    features = []
    for i in range(0, h - cell_size + 1, cell_size):
        for j in range(0, w - cell_size + 1, cell_size):
            cell_mag = magnitude[i:i + cell_size, j:j + cell_size].ravel()
            cell_ori = orientation[i:i + cell_size, j:j + cell_size].ravel()
            hist = np.zeros(n_bins, dtype=np.float32)
            for m, o in zip(cell_mag, cell_ori):
                b = int(o / bin_width)
                if b >= n_bins:
                    b = n_bins - 1
                hist[b] += m
            features.append(hist)

    features = np.concatenate(features)
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features.astype(np.float32)
