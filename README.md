# Reconnaissance d'automates finis à partir d'images

Reconstruit la structure d'un DFA (états, initial, acceptants, alphabet,
transitions) depuis une image Graphviz, et l'exporte en table textuelle :

```
#states / #initial / #accepting / #alphabet / #transitions
```
Transitions au format `source:symbole>destination`, une par ligne.

Les noms des états (`s0`, `s1`, ... `s12`) sont lus directement dans
l'image, caractère par caractère, et repris tels quels dans la table ; si
un nom est illisible ou en double, l'état reçoit un nom d'indice libre.

Deux implémentations :

- `python/` — prototype de référence (`numpy`/`OpenCV`), qui porte aussi
  l'entraînement des k-NN, la génération du corpus et l'évaluation ;
- `cpp/` — portage de l'inférence (image → table), ~8× plus rapide.

Elles partagent deux modèles entraînés : `data/knn_letters.bin` (lettres,
pour les symboles de transitions et le `s` des noms) et
`data/knn_digits.bin` (chiffres des noms). Sur un corpus de 150 images
(jusqu'à 12 états), leurs tables sont identiques octet pour octet ; les
noms d'états sont tous lus correctement, 70/150 automates sont
parfaitement reconstruits noms compris, et en moyenne 94,7 % de la
structure est correcte (les erreurs restantes viennent du suivi des
flèches sur les gros automates denses).

## Python

```bash
cd python
uv run main.py                       # corpus + entraînement + évaluation
uv run src/automaton_generator.py    # générer le corpus d'images (data/base_automata)
uv run src/main.py image.png         # analyser une image
uv run src/batch_eval.py ../data/base_automata/   # scores sur le corpus
```

## C++

Dépendance : OpenCV (`brew install opencv`).

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build                        # tests
./cpp/build/pyplus image.png [--table sortie.txt]
./cpp/build/pyplus --batch data/base_automata --out predites/
```

Les modèles se passent avec `--letters` et `--digits` (défaut :
`data/knn_letters.bin` et `data/knn_digits.bin`, côté Python comme C++).

Les tests `*_cross` comparent chaque étape (k-NN, features, segmentation,
tables) aux sorties du prototype sur des images réelles ; les fixtures se
régénèrent avec les scripts `cpp/tests/gen_*.py`.

## Performances

Comparaison du temps d'exécution en traitement par lots sur le corpus `data/base_automata` (150 images d'automates) :

| Version | Temps total |
|---------|-------------|
| **C++** | **~4.2 s**  |
| **Python** | **~33.5 s** |

Le portage en C++ (`pyplus`) permet une accélération par un facteur 8 de l'inférence (segmentation et reconnaissance) par rapport au script Python de référence (`batch_eval.py`).
