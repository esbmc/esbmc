#define metasmt_convt stp_metasmt_convt
#define metasmt_smt_ast stp_metasmt_smt_ast
#define metasmt_smt_sort stp_metasmt_smt_sort
#define Lookup stp_metasmt_Lookup
#define SOLVER_TYPE metaSMT::solver::STP

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>

#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/backend/STP.hpp>
#include <metaSMT/backend/SAT_Clause.hpp>
#include <metaSMT/BitBlast.hpp>

#include "metasmt_conv.cpp"

// To avoid having to build metaSMT into multiple files,
prop_convt *
create_new_metasmt_stp_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns)
{
  return new stp_metasmt_convt(int_encoding, is_cpp, ns);
}

const std::string
stp_metasmt_convt::solver_text()
{
  return "STP";
}
