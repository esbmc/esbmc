#include "solve.h"

#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/smt_tuple_flat.h>

typedef smt_convt *(solver_creator)
  (bool int_encoding, const namespacet &ns, bool is_cpp, const optionst &opts,
   tuple_iface **tuple_api, array_iface **array_api);

typedef smt_convt *(*solver_creator_ptr)
  (bool int_encoding, const namespacet &ns, bool is_cpp, const optionst &opts,
   tuple_iface **tuple_api, array_iface **array_api);

solver_creator create_new_smtlib_solver;
solver_creator create_new_z3_solver;
solver_creator create_new_minisat_solver;
solver_creator create_new_boolector_solver;
solver_creator create_new_cvc_solver;
solver_creator create_new_mathsat_solver;

struct solver_config {
  std::string name;
  solver_creator_ptr create;
};

static struct solver_config solvers[] =  {
  { "smtlib", create_new_smtlib_solver },
#ifdef Z3
  { "z3", create_new_z3_solver },
#endif
#ifdef MINISAT
  { "minisat", create_new_minisat_solver },
#endif
#ifdef BOOLECTOR
  { "boolector", create_new_boolector_solver },
#endif
#ifdef USECVC
  { "cvc", create_new_cvc_solver },
#endif
#ifdef MATHSAT
  { "mathsat", create_new_mathsat_solver }
#endif
};

static const std::string list_of_all_solvers[] =
{ "z3", "smtlib", "minisat", "boolector", "mathsat", "cvc"};

static const unsigned int total_num_of_solvers =
sizeof(list_of_all_solvers) / sizeof(std::string);

static const unsigned int num_solvers =
sizeof(solvers) / sizeof(solver_config);

static smt_convt *
create_solver(std::string the_solver,
            bool is_cpp, bool int_encoding, const namespacet &ns,
            const optionst &options, tuple_iface **tuple_api,
            array_iface **array_api)
{

  for (unsigned int i = 0; i < num_solvers; i++) {
    if (the_solver == solvers[i].name) {
      return solvers[i].create(int_encoding, ns, is_cpp,
                               options, tuple_api, array_api);
    }
  }

  std::cerr << "The " << the_solver << " solver has not been built into this version of ESBMC, sorry" << std::endl;
  abort();
}

static smt_convt *
pick_solver(bool is_cpp, bool int_encoding, const namespacet &ns,
            const optionst &options, tuple_iface **tuple_api,
            array_iface **array_api)
{
  unsigned int i;
  std::string the_solver = "";

  *tuple_api = NULL;
  *array_api = NULL;

  for (i = 0; i < total_num_of_solvers; i++) {
    if (options.get_bool_option(list_of_all_solvers[i])) {
      if (the_solver != "") {
        std::cerr << "Please only specify one solver" << std::endl;
        abort();
      }

      the_solver = list_of_all_solvers[i];
    }
  }

  if (the_solver == "") {
#ifdef Z3
    std::cerr << "No solver specified; defaulting to Z3" << std::endl;
    the_solver = "z3";
#else
    std::cerr << "No solver specified and Z3 is not enabled: please specify a solver" << std::endl;
    abort();
#endif
  }

  return create_solver(the_solver, is_cpp, int_encoding, ns,
                       options, tuple_api, array_api);
}

smt_convt *
create_solver_factory1(const std::string &solver_name, bool is_cpp,
                       bool int_encoding, const namespacet &ns,
                       const optionst &options,
                       tuple_iface **tuple_api,
                       array_iface **array_api)
{
  if (solver_name == "")
    // Pick one based on options.
    return pick_solver(is_cpp, int_encoding, ns, options, tuple_api, array_api);

  *tuple_api = NULL;
  *array_api = NULL;

  return create_solver(solver_name, is_cpp, int_encoding, ns,
                       options, tuple_api, array_api);
}


smt_convt *
create_solver_factory(const std::string &solver_name, bool is_cpp,
                      bool int_encoding, const namespacet &ns,
                      const optionst &options)
{
  tuple_iface *tuple_api = NULL;
  array_iface *array_api = NULL;
  smt_convt *ctx = create_solver_factory1(solver_name, is_cpp, int_encoding, ns, options, &tuple_api, &array_api);

  bool node_flat = options.get_bool_option("tuple-node-flattener");
  bool sym_flat = options.get_bool_option("tuple-sym-flattener");

  // Pick a tuple flattener to use. If the solver has native support, and no
  // options were given, use that by default
  if (tuple_api != NULL && !node_flat && !sym_flat)
    ctx->set_tuple_iface(tuple_api);
  // Use the node flattener if specified
  else if (node_flat)
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));
  // Use the symbol flattener if specified
  else if (sym_flat)
    ctx->set_tuple_iface(new smt_tuple_sym_flattener(ctx, ns));
  // Default: node flattener
  else
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));

  ctx->smt_post_init();
  return ctx;
}
