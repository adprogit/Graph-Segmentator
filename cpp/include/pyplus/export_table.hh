// Export du resultat de segmentation vers le format table, port de
// python/src/export_table.py (et de to_table de automaton_parser.py).
//
// Le format :
//     #states / #initial / #accepting / #alphabet / #transitions
//     transitions au format src:sym>dst
//
// Le parsing des tables et la comparaison restent en Python
// (automaton_parser.py, automaton_compare.py : outillage d'evaluation).

#pragma once

#include <optional>
#include <set>
#include <string>
#include <tuple>

#include "pyplus/pipeline.hh"

namespace pyplus {

struct Automaton {
  std::set<std::string> states;
  std::optional<std::string> initial;
  std::set<std::string> accepting;
  std::set<std::string> alphabet;
  std::set<std::tuple<std::string, std::string, std::string>> transitions;
};

// Serialise un Automaton au format table sectionne (sections triees).
std::string to_table(const Automaton &aut);

// Convertit le resultat de segment_automaton en une chaine au format table.
// Les etiquettes sans symbole reconnu sont ignorees (on n'invente pas).
std::string result_to_table(const AutomatonResult &result);

// Ecrit la table reconstruite sur disque.
void save_table(const AutomatonResult &result, const std::string &path);

} // namespace pyplus
