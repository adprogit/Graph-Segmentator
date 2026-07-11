#include "pyplus/export_table.hh"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pyplus {

std::string to_table(const Automaton &aut) {
  // std::set itere en ordre trie, comme sorted() dans le prototype
  std::ostringstream out;
  out << "#states\n";
  for (const std::string &s : aut.states)
    out << s << "\n";
  out << "#initial\n";
  out << (aut.initial ? *aut.initial : "") << "\n";
  out << "#accepting\n";
  for (const std::string &s : aut.accepting)
    out << s << "\n";
  out << "#alphabet\n";
  for (const std::string &s : aut.alphabet)
    out << s << "\n";
  out << "#transitions\n";
  for (const auto &[src, sym, dst] : aut.transitions)
    out << src << ":" << sym << ">" << dst << "\n";
  return out.str();
}

std::string result_to_table(const AutomatonResult &result) {
  Automaton aut;
  const std::size_t n = result.states.size();
  // nom reconnu dans l'image si present, repli d'indice sinon
  std::vector<std::string> state_names(n);
  for (std::size_t i = 0; i < n; ++i) {
    state_names[i] = result.states[i].name.empty() ? "s" + std::to_string(i)
                                                   : result.states[i].name;
    aut.states.insert(state_names[i]);
  }

  // alphabet + transitions, deduits des symboles reconnus
  for (const Arrow &edge : result.arrows) {
    const std::string &src = state_names[edge.source];
    const std::string &dst = state_names[edge.dest];
    for (const ArrowLabel &label : edge.labels) {
      if (!label.symbol)
        continue; // non reconnu -> on n'invente pas
      aut.alphabet.insert(*label.symbol);
      aut.transitions.insert({src, *label.symbol, dst});
    }
  }

  if (result.initial)
    aut.initial = state_names[*result.initial];
  else if (n > 0)
    aut.initial = state_names[0]; // comme le prototype (pas le min du set)

  for (std::size_t i = 0; i < n; ++i)
    if (result.states[i].accepting)
      aut.accepting.insert(state_names[i]);

  return to_table(aut);
}

void save_table(const AutomatonResult &result, const std::string &path) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("impossible d'ecrire : " + path);
  f << result_to_table(result);
}

} // namespace pyplus
