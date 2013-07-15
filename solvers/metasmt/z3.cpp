#define metasmt_convt z3_metasmt_convt
#define metasmt_smt_ast z3_metasmt_smt_ast
#define metasmt_smt_sort z3_metasmt_smt_sort
#define Lookup z3_metasmt_Lookup
#define SOLVER_TYPE metaSMT::solver::Z3_Backend

#include <solvers/smt/smt_conv.h>

#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/backend/Z3_Backend.hpp>

#include "metasmt_conv.cpp"

// To avoid having to build metaSMT into multiple files,
prop_convt *
create_new_metasmt_z3_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns)
{
  return new z3_metasmt_convt(int_encoding, is_cpp, ns);
}

const std::string
z3_metasmt_convt::solver_text()
{
  unsigned int major, minor, build, revision;
  Z3_get_version(&major, &minor, &build, &revision);
  std::stringstream ss;
  ss << "Z3 v" << major << "." << minor;
  return ss.str();
}
