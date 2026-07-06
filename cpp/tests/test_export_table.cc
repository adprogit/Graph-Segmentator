// Test unitaire de l'export table, sur le meme resultat simule que le
// __main__ de python/src/export_table.py (sortie attendue capturee depuis
// le prototype).

#include <iostream>
#include <string>

#include "pyplus/export_table.hh"

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " echec : " #cond "\n";      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

pyplus::Arrow make_arrow(int src, int dst, const std::string &symbol) {
  pyplus::Arrow edge;
  edge.source = src;
  edge.dest = dst;
  pyplus::ArrowLabel label;
  label.symbol = symbol;
  edge.labels.push_back(label);
  return edge;
}

} // namespace

int main() {
  pyplus::AutomatonResult result;
  result.states.resize(2);
  result.states[1].accepting = true;
  result.initial = 0;
  result.arrows = {
      make_arrow(0, 0, "b"),
      make_arrow(0, 1, "d"),
      make_arrow(1, 1, "b"),
      make_arrow(1, 0, "d"),
  };

  const std::string expected = "#states\ns0\ns1\n#initial\ns0\n#accepting\ns1\n"
                               "#alphabet\nb\nd\n#transitions\n"
                               "s0:b>s0\ns0:d>s1\ns1:b>s1\ns1:d>s0\n";
  CHECK(pyplus::result_to_table(result) == expected);

  // etiquette non reconnue -> ignoree
  pyplus::ArrowLabel unrecognized;
  result.arrows[0].labels.push_back(unrecognized);
  CHECK(pyplus::result_to_table(result) == expected);

  // resultat vide : initial absent -> ligne vide
  const pyplus::AutomatonResult empty;
  CHECK(pyplus::result_to_table(empty) ==
        "#states\n#initial\n\n#accepting\n#alphabet\n#transitions\n");

  // pas d'initial detecte mais des etats -> s0 par defaut
  pyplus::AutomatonResult no_initial;
  no_initial.states.resize(2);
  CHECK(pyplus::result_to_table(no_initial) ==
        "#states\ns0\ns1\n#initial\ns0\n#accepting\n#alphabet\n"
        "#transitions\n");

  std::cout << "export_table OK\n";
  return 0;
}
