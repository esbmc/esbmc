#define metasmt_convt boolector_metasmt_convt
#define metasmt_smt_ast boolector_metasmt_smt_ast
#define metasmt_smt_sort boolector_metasmt_smt_sort
#define Lookup boolector_metasmt_Lookup
#define SOLVER_TYPE metaSMT::solver::Boolector

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>

static void boolector_abort_function(void *bees __attribute__((unused))) {
  return;
}

#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/backend/Boolector.hpp>
#include <metaSMT/backend/SAT_Clause.hpp>
#include <metaSMT/BitBlast.hpp>

#include "metasmt_conv.cpp"

// To avoid having to build metaSMT into multiple files,
smt_convt *
create_new_metasmt_boolector_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns)
{
  return new boolector_metasmt_convt(int_encoding, is_cpp, ns);
}

const std::string
boolector_metasmt_convt::solver_text()
{
  return "Boolector";
}
