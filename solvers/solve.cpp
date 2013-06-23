#include "solve.h"

#include <solvers/z3/z3_conv.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/metasmt/metasmt_conv.h>

prop_convt *
create_solver_factory(const std::string &solver_name, bool is_cpp,
                      bool int_encoding, const namespacet &ns,
                      const optionst &options)
{
  if (solver_name == "z3") {
    return new z3_convt(int_encoding, is_cpp, ns);
  } else if (solver_name == "smtlib") {
    return new smtlib_convt(int_encoding, ns, is_cpp, options);
  } else if (solver_name == "metasmt") {
    return new metasmt_convt(int_encoding, is_cpp, ns);
  } else {
    std::cerr << "Unrecognized solver \"" << solver_name << "\" created"
              << std::endl;
    abort();
  }
}
