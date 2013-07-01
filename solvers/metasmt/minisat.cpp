#define metasmt_convt minisat_metasmt_convt
#define metasmt_smt_ast minisat_metasmt_smt_ast
#define metasmt_smt_sort minisat_metasmt_smt_sort
#define Lookup minisat_metasmt_Lookup
#define SOLVER_TYPE metaSMT::BitBlast < metaSMT::SAT_Clause< metaSMT::solver::MiniSAT > >

#include "metasmt_conv.cpp"

// To avoid having to build metaSMT into multiple files,
prop_convt *
create_new_metasmt_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
  return new minisat_metasmt_convt(int_encoding, is_cpp, ns);
}

