#include <solve.h>
#include <solver_config.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_tuple_flat.h>

solver_creator create_new_smtlib_solver;
solver_creator create_new_z3_solver;
solver_creator create_new_minisat_solver;
solver_creator create_new_boolector_solver;
solver_creator create_new_cvc_solver;
solver_creator create_new_mathsat_solver;
solver_creator create_new_yices_solver;

const struct esbmc_solver_config esbmc_solvers[] =  {
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
  { "mathsat", create_new_mathsat_solver },
#endif
#ifdef YICES
  { "yices", create_new_yices_solver }
#endif
};

const std::string list_of_all_solvers[] =
{ "z3", "smtlib", "minisat", "boolector", "mathsat", "cvc", "yices"};

const unsigned int total_num_of_solvers =
sizeof(list_of_all_solvers) / sizeof(std::string);

const unsigned int esbmc_num_solvers =
sizeof(esbmc_solvers) / sizeof(esbmc_solver_config);

static smt_convt *
create_solver(const std::string&& the_solver,
            bool int_encoding, const namespacet &ns,
            const optionst &options, tuple_iface **tuple_api,
            array_iface **array_api)
{

  for (const auto & esbmc_solver : esbmc_solvers) {
    if (the_solver == esbmc_solver.name) {
      return esbmc_solver.create(int_encoding, ns,
                               options, tuple_api, array_api);
    }
  }

  std::cerr << "The " << the_solver << " solver has not been built into this version of ESBMC, sorry" << std::endl;
  abort();
}

static const std::string
pick_default_solver()
{

#ifdef BOOLECTOR
  std::cerr << "No solver specified; defaulting to Boolector" << std::endl;
  return "boolector";
#else
  // Pick whatever's first in the list.
  if (esbmc_num_solvers == 1) {
    std::cerr << "No solver backends built into ESBMC; please either build ";
    std::cerr << "some in, or explicitly configure the smtlib backend";
    std::cerr << std::endl;
    abort();
  } else {
    std::cerr << "No solver specified; defaulting to " << esbmc_solvers[1].name;
    std::cerr << std::endl;
    return esbmc_solvers[1].name;
  }
#endif
}

static smt_convt *
pick_solver(bool int_encoding, const namespacet &ns,
            const optionst &options, tuple_iface **tuple_api,
            array_iface **array_api)
{
  unsigned int i;
  std::string the_solver;

  for (i = 0; i < total_num_of_solvers; i++) {
    if (options.get_bool_option(list_of_all_solvers[i])) {
      if (the_solver != "") {
        std::cerr << "Please only specify one solver" << std::endl;
        abort();
      }

      the_solver = list_of_all_solvers[i];
    }
  }

  if (the_solver == "")
    the_solver = pick_default_solver();

  return create_solver(std::move(the_solver), int_encoding, ns,
                       options, tuple_api, array_api);
}

smt_convt *
create_solver_factory1(const std::string &solver_name,
                       bool int_encoding, const namespacet &ns,
                       const optionst &options,
                       tuple_iface **tuple_api,
                       array_iface **array_api)
{
  if (solver_name == "")
    // Pick one based on options.
    return pick_solver(int_encoding, ns, options, tuple_api, array_api);

  return create_solver(std::move(solver_name), int_encoding, ns,
                       options, tuple_api, array_api);
}


smt_convt *
create_solver_factory(const std::string &solver_name,
                      bool int_encoding, const namespacet &ns,
                      const optionst &options)
{
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  smt_convt *ctx = create_solver_factory1(solver_name, int_encoding, ns, options, &tuple_api, &array_api);

  bool node_flat = options.get_bool_option("tuple-node-flattener");
  bool sym_flat = options.get_bool_option("tuple-sym-flattener");
  bool array_flat = options.get_bool_option("array-flattener");

  // Pick a tuple flattener to use. If the solver has native support, and no
  // options were given, use that by default
  if (tuple_api != nullptr && !node_flat && !sym_flat)
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

  // Pick an array flattener to use. Again, pick the solver native one by
  // default, or the one specified, or if none of the above then use the built
  // in arrays -> to BV flattener.
  if (array_api != nullptr && !array_flat)
    ctx->set_array_iface(array_api);
  else if (array_flat)
    ctx->set_array_iface(new array_convt(ctx));
  else
    ctx->set_array_iface(new array_convt(ctx));

  ctx->smt_post_init();
  return ctx;
}
