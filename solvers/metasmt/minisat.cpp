#define metasmt_convt minisat_metasmt_convt
#define metasmt_smt_ast minisat_metasmt_smt_ast
#define metasmt_smt_sort minisat_metasmt_smt_sort
#define Lookup minisat_metasmt_Lookup
#define SOLVER_TYPE metaSMT::BitBlast < metaSMT::SAT_Clause< metaSMT::solver::MiniSAT > >

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>

#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/backend/MiniSAT.hpp>
#include <metaSMT/backend/SAT_Clause.hpp>
#include <metaSMT/BitBlast.hpp>

#define SOLVER_BITBLAST_ARRAYS

#include "metasmt_conv.cpp"

// To avoid having to build metaSMT into multiple files,
smt_convt *
create_new_metasmt_minisat_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns)
{
  return new minisat_metasmt_convt(int_encoding, is_cpp, ns);
}

const std::string
minisat_metasmt_convt::solver_text()
{
  return "MiniSAT";
}
