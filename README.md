# Reconnaissance d'automates finis à partir d'images

Reconstruit la structure d'un DFA (états, initial, acceptants, alphabet,
transitions) depuis une image Graphviz, et l'exporte en table textuelle.
Deux implémentations : un prototype Python (`numpy`/`OpenCV`, référence) et
un portage C++ en cours.

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
    ├── automaton_compare.py   Jaccard, isomorphisme, appariement optimal
    ├── automaton_render.py    automate → DOT → rendu Graphviz
    ├── automaton_generator.py génération du corpus (3 niveaux)
    └── batch_eval.py      évaluation batch : scores par niveau + échecs JSON

cpp/                       portage C++ (inférence seule)
├── CMakeLists.txt
├── src/                   .cc, mêmes noms que les modules Python
├── include/pyplus/        en-têtes publics
└── tests/                 tests unitaires

data/                      partagé entre les deux implémentations
├── knn_model.bin          modèle k-NN entraîné (format binaire portable)
└── base_automata/         corpus généré (non versionné)
```

Le portage C++ ne couvre que l'inférence (segmentation → features →
classifier → export_table + lecture de `model_io`). L'entraînement, la
génération du corpus et l'évaluation restent en Python. Le format de table
étant textuel, les deux implémentations se valident en comparant leurs
sorties sur les mêmes images de `data/base_automata/`.

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

Portage de l'inférence complet : `model_io`, `classifier`, `features`,
`segmentation`, `export_table` et le pipeline de bout en bout (image →
table). L'entraînement, la génération du corpus et l'évaluation restent
en Python. Dépendance : OpenCV (`brew install opencv`) ; sans OpenCV,
seuls `model_io` et `classifier` compilent.

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
