// Chaine complete de segmentation + reconnaissance d'une image d'automate,
// port de segment_automaton (python/src/main.py).

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "pyplus/classifier.hh"
#include "pyplus/segmentation.hh"

namespace pyplus {

struct AutomatonResult {
  std::vector<State> states;
  std::vector<std::vector<int>> matrix; // index dans arrows, ou -1
  std::vector<Arrow> arrows;
  std::optional<int> initial;
};

// Execute toute la chaine sur une image d'automate ; si classifier est
// non nul, reconnait les symboles des etiquettes.
// Lance std::runtime_error si l'image est introuvable.
AutomatonResult segment_automaton(const std::string &image_path,
                                  const KNNClassifier *classifier = nullptr);

} // namespace pyplus
