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
  // meme resultat simule que le __main__ Python : noms reconnus non
  // contigus (s3, s7), repris tels quels dans la table
  pyplus::AutomatonResult result;
  result.states.resize(2);
  result.states[0].name = "s3";
  result.states[1].name = "s7";
  result.states[1].accepting = true;
  result.initial = 0;
  result.arrows = {
      make_arrow(0, 0, "b"),
      make_arrow(0, 1, "d"),
      make_arrow(1, 1, "b"),
      make_arrow(1, 0, "d"),
  };

  const std::string expected = "#states\ns3\ns7\n#initial\ns3\n#accepting\ns7\n"
                               "#alphabet\nb\nd\n#transitions\n"
                               "s3:b>s3\ns3:d>s7\ns7:b>s7\ns7:d>s3\n";
  CHECK(pyplus::result_to_table(result) == expected);

  // etiquette non reconnue -> ignoree
  pyplus::ArrowLabel unrecognized;
  result.arrows[0].labels.push_back(unrecognized);
  CHECK(pyplus::result_to_table(result) == expected);

  // sans noms reconnus -> repli sur le nommage par indice
  pyplus::AutomatonResult unnamed;
  unnamed.states.resize(2);
  unnamed.states[1].accepting = true;
  unnamed.initial = 0;
  unnamed.arrows = {make_arrow(0, 1, "d")};
  CHECK(pyplus::result_to_table(unnamed) ==
        "#states\ns0\ns1\n#initial\ns0\n#accepting\ns1\n"
        "#alphabet\nd\n#transitions\ns0:d>s1\n");

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
