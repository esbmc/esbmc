#include "solve.h"

#include <solvers/z3/z3_conv.h>
#include <solvers/smtlib/smtlib_conv.h>

// For the purpose of vastly reducing build times:
prop_convt *
create_new_metasmt_minisat_solver(bool int_encoding, bool is_cpp,
                                  const namespacet &ns);
prop_convt *
create_new_metasmt_z3_solver(bool int_encoding, bool is_cpp,
                             const namespacet &ns);
prop_convt *
create_new_metasmt_boolector_solver(bool int_encoding, bool is_cpp,
                                    const namespacet &ns);

static prop_convt *
create_z3_solver(bool is_cpp, bool int_encoding, const namespacet &ns)
{
#ifndef Z3
    std::cerr << "Sorry, Z3 support was not built into this version of ESBMC"
              << std::endl;
    abort();
#else
    return new z3_convt(int_encoding, is_cpp, ns);
#endif
}

static prop_convt *
create_metasmt_minisat_solver(bool is_cpp, bool int_encoding,
                              const namespacet &ns)
{
#if !defined(METASMT) || !defined(MINISAT)
    std::cerr << "Sorry, metaSMT minisat support was not built into this "
                 "version of " << "ESBMC" << std::endl;
    abort();
#else
    return create_new_metasmt_minisat_solver(int_encoding, is_cpp, ns);
#endif
}

static prop_convt *
create_metasmt_z3_solver(bool is_cpp, bool int_encoding, const namespacet &ns)
{
#if !defined(METASMT) || !defined(Z3)
    std::cerr << "Sorry, metaSMT Z3 support was not built into this version of "
              << "ESBMC" << std::endl;
    abort();
#else
    return create_new_metasmt_z3_solver(int_encoding, is_cpp, ns);
#endif
}

static prop_convt *
create_metasmt_boolector_solver(bool is_cpp, bool int_encoding,
                                const namespacet &ns)
{
#if !defined(METASMT) || !defined(BOOLECTOR)
    std::cerr << "Sorry, metaSMT Boolector support was not built into this "
              << "version of ESBMC" << std::endl;
    abort();
#else
    return create_new_metasmt_boolector_solver(int_encoding, is_cpp, ns);
#endif
}

static const unsigned int num_of_solvers = 5;
static const std::string list_of_solvers[] =
{ "z3", "smtlib", "minisat", "metasmt", "boolector" };

static prop_convt *
pick_solver(bool is_cpp, bool int_encoding, const namespacet &ns,
            const optionst &options)
{
  unsigned int i, total_solvers = 0;
  for (i = 0; i < num_of_solvers; i++)
    total_solvers += (options.get_bool_option(list_of_solvers[i])) ? 1 : 0;

  if (total_solvers == 0) {
    std::cerr << "No solver specified; defaulting to Z3" << std::endl;
  } else if (total_solvers > 1) {
    // Metasmt is one fewer solver.
    if (options.get_bool_option("metasmt") && total_solvers == 2) {
      ;
    } else {
      std::cerr << "Please only specify one solver" << std::endl;
      abort();
    }
  }

  if (options.get_bool_option("smtlib")) {
    return new smtlib_convt(int_encoding, ns, is_cpp, options);
  } else if (options.get_bool_option("metasmt")) {
    if (options.get_bool_option("minisat")) {
      return create_metasmt_minisat_solver(is_cpp, int_encoding, ns);
    } else if (options.get_bool_option("z3")) {
      return create_metasmt_z3_solver(is_cpp, int_encoding, ns);
    } else if (options.get_bool_option("boolector")) {
      return create_metasmt_boolector_solver(is_cpp, int_encoding, ns);
    } else {
      std::cerr << "You must specify a backend solver when using the metaSMT "
                << "framework" << std::endl;
      abort();
    }
  } else {
    return create_z3_solver(is_cpp, int_encoding, ns);
  }
}

prop_convt *
create_solver_factory(const std::string &solver_name, bool is_cpp,
                      bool int_encoding, const namespacet &ns,
                      const optionst &options)
{
  if (solver_name == "")
    // Pick one based on options.
    return pick_solver(is_cpp, int_encoding, ns, options);

  if (solver_name == "z3") {
    return create_z3_solver(is_cpp, int_encoding, ns);
  } else if (solver_name == "smtlib") {
    return new smtlib_convt(int_encoding, ns, is_cpp, options);
  } else if (solver_name == "metasmt") {
    if (options.get_bool_option("minisat")) {
      return create_metasmt_minisat_solver(is_cpp, int_encoding, ns);
    } else {
      std::cerr << "You must specify a backend solver when using the metaSMT "
                << "framework" << std::endl;
      abort();
    }
  } else {
    std::cerr << "Unrecognized solver \"" << solver_name << "\" created"
              << std::endl;
    abort();
  }
}
