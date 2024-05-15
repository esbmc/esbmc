#include <solve.h>
#include <solver_config.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/fp/fp_conv.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/tuple/smt_tuple_node.h>
#include <solvers/smt/tuple/smt_tuple_sym.h>

#include <unordered_map>

solver_creator create_new_smtlib_solver;
solver_creator create_new_z3_solver;
solver_creator create_new_minisat_solver;
solver_creator create_new_boolector_solver;
solver_creator create_new_cvc_solver;
solver_creator create_new_cvc5_solver;
solver_creator create_new_mathsat_solver;
solver_creator create_new_yices_solver;
solver_creator create_new_bitwuzla_solver;

static const std::unordered_map<std::string, solver_creator *> esbmc_solvers = {
#ifdef SMTLIB
  {"smtlib", create_new_smtlib_solver},
#endif
#ifdef Z3
  {"z3", create_new_z3_solver},
#endif
#ifdef MINISAT
  {"minisat", create_new_minisat_solver},
#endif
#ifdef BOOLECTOR
  {"boolector", create_new_boolector_solver},
#endif
#ifdef USECVC
  {"cvc4", create_new_cvc_solver},
#endif
#ifdef USECVC5
  {"cvc5", create_new_cvc5_solver},
#endif
#ifdef MATHSAT
  {"mathsat", create_new_mathsat_solver},
#endif
#ifdef YICES
  {"yices", create_new_yices_solver},
#endif
#ifdef BITWUZLA
  {"bitwuzla", create_new_bitwuzla_solver}
#endif
};

static const std::string all_solvers[] = {
  "smtlib",
  "z3",
  "minisat",
  "boolector",
  "cvc4",
  "cvc5",
  "mathsat",
  "yices",
  "bitwuzla"};

static std::string pick_default_solver()
{
#ifdef BOOLECTOR
  log_status("No solver specified; defaulting to Boolector");
  return "boolector";
#else
  // Pick whatever's first in the list except for the smtlib solver
  for (const std::string &name : all_solvers)
  {
    if (name == "smtlib" || !esbmc_solvers.count(name))
      continue;
    log_status("No solver specified; defaulting to {}", name);
    return name;
  }
  log_error(
    "No solver backends built into ESBMC; please either build "
    "some in, or explicitly configure the smtlib backend");
  abort();
#endif
}

static solver_creator &
pick_solver(std::string &solver_name, const optionst &options)
{
  if (solver_name == "")
  {
    // Pick one based on options.
    for (const std::string &name : all_solvers)
      if (options.get_bool_option(name))
      {
        if (solver_name != "")
        {
          log_error("Please only specify one solver");
          abort();
        }

        solver_name = name;
      }
  }

  if (solver_name == "")
    solver_name = options.get_option("default-solver");

  if (solver_name == "")
    solver_name = pick_default_solver();

  auto it = esbmc_solvers.find(solver_name);
  if (it != esbmc_solvers.end())
    return *it->second;

  log_error(
    "The {} solver has not been built into this version of ESBMC, sorry",
    solver_name);
  abort();
}

smt_convt *create_solver(
  std::string solver_name,
  const namespacet &ns,
  const optionst &options)
{
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;

  solver_creator &factory = pick_solver(solver_name, options);
  smt_convt *ctx = factory(options, ns, &tuple_api, &array_api, &fp_api);

  bool node_flat = options.get_bool_option("tuple-node-flattener");
  bool sym_flat = options.get_bool_option("tuple-sym-flattener");
  bool array_flat = options.get_bool_option("array-flattener");
  bool fp_to_bv = options.get_bool_option("fp2bv");

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

  if (fp_api == nullptr || fp_to_bv)
    ctx->set_fp_conv(new fp_convt(ctx));
  else
    ctx->set_fp_conv(fp_api);

  ctx->smt_post_init();
  return ctx;
}
