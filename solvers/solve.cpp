#include "solve.h"

#include <solvers/z3/z3_conv.h>
#include <solvers/smtlib/smtlib_conv.h>

// For the purpose of vastly reducing build times:
prop_convt *
create_new_metasmt_minisat_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns);

prop_convt *
create_solver_factory(const std::string &solver_name, bool is_cpp,
                      bool int_encoding, const namespacet &ns,
                      const optionst &options)
{
  if (solver_name == "z3") {
#ifndef Z3
    std::cerr << "Sorry, Z3 support was not built into this version of ESBMC"
              << std::endl;
    abort();
#else
    return new z3_convt(int_encoding, is_cpp, ns);
#endif
  } else if (solver_name == "smtlib") {
    return new smtlib_convt(int_encoding, ns, is_cpp, options);
  } else if (solver_name == "metasmt") {
#ifndef METASMT
    std::cerr << "Sorry, metaSMT support was not built into this version of "
              << "ESBMC" << std::endl;
    abort();
#else
    return create_new_metasmt_minisat_solver(int_encoding, is_cpp, ns);
#endif
  } else {
    std::cerr << "Unrecognized solver \"" << solver_name << "\" created"
              << std::endl;
    abort();
  }
}
