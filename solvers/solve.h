#include <string>

#include <config.h>
#include <namespace.h>

#include <solvers/prop/prop_conv.h>

prop_convt *create_solver_factory(const std::string &solver_name, bool is_cpp,
                                  bool int_encoding, const namespacet &ns,
                                  const optionst &options);
