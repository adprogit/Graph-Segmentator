# Reconnaissance d'automates finis à partir d'images

Reconstruit la structure d'un DFA (états, initial, acceptants, alphabet,
transitions) depuis une image Graphviz, et l'exporte en table textuelle :

```
#states / #initial / #accepting / #alphabet / #transitions
```
Transitions au format `source:symbole>destination`, une par ligne.

Deux implémentations :

- `python/` — prototype de référence (`numpy`/`OpenCV`), qui porte aussi
  l'entraînement du k-NN, la génération du corpus et l'évaluation ;
- `cpp/` — portage de l'inférence (image → table), ~5× plus rapide.

Elles partagent le modèle entraîné `data/knn_model.bin`. Sur un corpus de
150 images, leurs tables sont identiques octet pour octet ; 80/150
automates sont parfaitement reconstruits et en moyenne 94,7 % des éléments
sont corrects (les noms d'états n'important pas).

## Python

```bash
cd python
uv run main.py                       # corpus + entraînement + évaluation
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

Les tests `*_cross` comparent chaque étape (k-NN, features, segmentation,
tables) aux sorties du prototype sur des images réelles ; les fixtures se
régénèrent avec les scripts `cpp/tests/gen_*.py`.
