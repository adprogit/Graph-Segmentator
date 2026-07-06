# Reconnaissance d'automates finis à partir d'images

Reconstruit la structure d'un DFA (états, initial, acceptants, alphabet,
transitions) depuis une image Graphviz, et l'exporte en table textuelle.
Deux implémentations : un prototype Python (`numpy`/`OpenCV`, référence) et
un portage C++ de l'inférence, validé octet pour octet contre le prototype
et ~5× plus rapide. L'entraînement, la génération du corpus et l'évaluation
restent en Python.

## Structure

```
python/                    prototype de référence
├── main.py                pipeline complet : corpus → entraînement → éval batch
├── pyproject.toml         dépendances (uv)
└── src/
    ├── main.py            segmentation + reconnaissance d'une image
    ├── segmentation.py    détection états, arcs (suivi de tracé), étiquettes
    ├── features.py        normalisation des crops + HOG maison (train/inférence)
    ├── classifier.py      classifieur k-NN maison
    ├── training.py        dataset synthétique (Graphviz) + entraînement
    ├── model_io.py        sérialisation binaire portable du modèle (.bin, C++)
    ├── export_table.py    résultat → format table + comparaison
    ├── automaton_parser.py    parsing table → objet Automaton
    ├── automaton_compare.py   similarité entre deux automates (noms d'états ignorés)
    ├── automaton_render.py    automate → DOT → rendu Graphviz
    ├── automaton_generator.py génération du corpus (3 niveaux)
    └── batch_eval.py      évaluation batch : scores par niveau + échecs JSON

cpp/                       portage C++ (inférence : image → table)
├── CMakeLists.txt
├── include/pyplus/        en-têtes publics
├── src/                   mêmes noms que les modules Python
│   ├── main.cc            CLI : une image, ou --batch sur un corpus
│   ├── pipeline.cc        chaîne complète (segment_automaton)
│   ├── segmentation.cc / features.cc / classifier.cc
│   └── model_io.cc / export_table.cc
└── tests/                 tests unitaires + validation croisée vs Python
    ├── gen_*.py           générateurs de fixtures (exécutent le prototype)
    └── fixtures/          entrées + sorties attendues, versionnées

data/                      partagé entre les deux implémentations
├── knn_model.bin          modèle k-NN entraîné (format binaire portable)
└── base_automata/         corpus généré (non versionné)
```

## Format de table

```
#states / #initial / #accepting / #alphabet / #transitions
```
Transitions au format `source:symbole>destination`, une par ligne.

## Utilisation — Python

Pipeline complet (à faire en premier : corpus + modèle) :
```bash
cd python
uv run main.py
```

Analyser une image :
```bash
uv run src/main.py chemin/vers/image.png
uv run src/main.py chemin/vers/image.png --debug
```

Évaluation batch sur le corpus :
```bash
uv run src/batch_eval.py ../data/base_automata/ --csv scores.csv
```

Le modèle est lu/écrit dans `data/knn_model.bin` par défaut, quel que soit
le répertoire courant.

## Utilisation — C++

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build          # tests, dont validation croisée vs Python
./cpp/build/pyplus chemin/vers/image.png [--table sortie.txt]
./cpp/build/pyplus --batch data/base_automata --out predites/   # corpus entier
```

Dépendance : OpenCV (`brew install opencv`) ; sans OpenCV, seuls
`model_io` et `classifier` compilent.

Les tests `*_cross` vérifient que le C++ reproduit les sorties du prototype :
prédictions kNN sur `data/knn_model.bin`, crops normalisés et vecteurs HOG
sur des glyphes réels, structure complète (états, initial, acceptants,
arcs, symboles) et tables finales sur des images d'automates. Les fixtures
se régénèrent avec :
```bash
uv run --project python cpp/tests/gen_cross_fixture.py
uv run --project python cpp/tests/gen_features_fixture.py   # graphviz requis
uv run --project python cpp/tests/gen_seg_fixture.py        # graphviz requis
```

## Parité et performance

Sur un corpus généré de 150 images (50 par niveau), les deux
implémentations produisent des tables **identiques octet pour octet**
(150/150), donc la même précision face à l'automate de référence :
80/150 automates parfaitement reconstruits, et en moyenne 94,7 % des
éléments corrects (états, initial, acceptants, alphabet, transitions —
les noms d'états n'important pas, seuls comptent la structure et les
symboles).

Temps du pipeline par image (modèle chargé une fois, Apple Silicon) :

| niveau | Python | C++ | gain |
|---|---|---|---|
| simple (2-3 états) | 4,1 ms | 1,5 ms | 2,8× |
| medium (3-4 états) | 12,5 ms | 3,3 ms | 3,7× |
| hard (5-6 états) | 60,7 ms | 10,1 ms | 5,9× |
| corpus complet | 3,9 s | 0,75 s | 5,2× |

Le gain croît avec la complexité : les boucles pixel à pixel (suivi de
tracé, histogrammes HOG) dominent sur les grandes images, alors que les
primitives OpenCV/numpy du prototype étaient déjà natives.
