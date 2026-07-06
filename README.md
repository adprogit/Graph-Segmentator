# Reconnaissance d'automates finis à partir d'images

Reconstruit la structure d'un DFA (états, initial, acceptants, alphabet,
transitions) depuis une image Graphviz, et l'exporte en table textuelle.
Pipeline en `numpy`/`OpenCV`, sans dépendance de haut niveau (portage C++ visé).

## Structure

```
main.py            point d'entrée
pyproject.toml     dépendances (uv)
src/
├── segmentation.py        détection états, arcs (suivi de tracé), étiquettes
├── features.py            normalisation des crops + HOG maison (train/inférence)
├── classifier.py          classifieur k-NN maison
├── training.py            dataset synthétique (Graphviz) + entraînement
├── model_io.py            sérialisation binaire portable du modèle (.bin, C++)
├── export_table.py        résultat → format table + comparaison
├── automaton_parser.py    parsing table → objet Automaton
├── automaton_compare.py   Jaccard, isomorphisme, appariement optimal des états
├── automaton_render.py    automate → DOT → rendu Graphviz
├── automaton_generator.py génération du corpus (3 niveaux)
├── batch_eval.py          évaluation batch : scores par niveau + échecs JSON
└── knn_model.bin          modèle entraîné
```

## Format de table

```
#states / #initial / #accepting / #alphabet / #transitions
```
Transitions au format `source:symbole>destination`, une par ligne.

## Utilisation

Pipeline complete (a faire en premier, genere la base) :
```bash
uv run src/training.py
```

Analyser une image :
```bash
uv run main.py chemin/vers/image.png
uv run main.py chemin/vers/image.png --model src/knn_model.bin --debug
```

Évaluation batch sur le corpus :
```bash
uv run src/batch_eval.py base_automata/ --csv scores.csv --failures failures.json
```
