import numpy as np


class KNNClassifier:
    """
    k plus proches voisins, implementation maison (numpy pur).
    Pensee pour un port C++ ligne a ligne : pas de dependance ML externe.
    """

    def __init__(self, k=3, weighted=False, reject_threshold=None):
        """
        Args:
            k: nombre de voisins consideres
            weighted: si True, vote pondere par 1/distance (departage les egalites)
            reject_threshold: si la distance min depasse ce seuil, retourne None
                              (crop parasite / hors alphabet). None = pas de rejet.
        """
        self.k = k
        self.weighted = weighted
        self.reject_threshold = reject_threshold
        self.X_train = None   # (N, D) vecteurs de features
        self.y_train = None   # (N,)   labels

    def fit(self, X, y):
        """Memorise les donnees d'entrainement (k-NN n'a pas de vrai training)."""
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y)

    def _squared_distances(self, x):
        """Distances au carre entre x (D,) et tous les points (sqrt inutile)."""
        return ((self.X_train - x) ** 2).sum(axis=1)

    def _k_nearest_indices(self, distances):
        """Indices des k plus proches points (pas forcement tries)."""
        k = min(self.k, len(distances))
        return np.argpartition(distances, k - 1)[:k]

    def _vote(self, neighbor_indices, distances):
        """
        Determine le label a partir des voisins.
        - vote simple : label majoritaire
        - vote pondere (self.weighted) : chaque voisin pese 1/distance
        """
        if self.weighted:
            weights = {}
            for idx in neighbor_indices:
                label = self.y_train[idx]
                weights[label] = weights.get(label, 0.0) + 1.0 / (distances[idx] + 1e-8)
            return max(weights, key=weights.get)
        labels = self.y_train[neighbor_indices]
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

    def predict_one(self, x):
        """Predit le label d'un vecteur x (D,), ou None si rejet."""
        distances = self._squared_distances(x)
        if self.reject_threshold is not None and distances.min() > self.reject_threshold ** 2:
            return None
        idx = self._k_nearest_indices(distances)
        return self._vote(idx, distances)

    def predict(self, X):
        """Predit pour un vecteur unique (D,) ou un lot (M, D)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            return self.predict_one(X)
        return np.array([self.predict_one(x) for x in X])
