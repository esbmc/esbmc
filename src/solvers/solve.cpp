#include <solve.h>
#include <solver_config.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/fp/fp_conv.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/tuple/smt_tuple_node.h>
#include <solvers/smt/tuple/smt_tuple_sym.h>

solver_creator create_new_smtlib_solver;
solver_creator create_new_z3_solver;
solver_creator create_new_minisat_solver;
solver_creator create_new_boolector_solver;
solver_creator create_new_cvc_solver;
solver_creator create_new_mathsat_solver;
solver_creator create_new_yices_solver;
solver_creator create_new_bitwuzla_solver;

const struct esbmc_solver_config esbmc_solvers[] = {
  {"smtlib", create_new_smtlib_solver},
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
  {"cvc", create_new_cvc_solver},
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

const std::string list_of_all_solvers[] = {
  "z3",
  "smtlib",
  "minisat",
  "boolector",
  "mathsat",
  "cvc",
  "yices",
  "bitwuzla"};

const unsigned int total_num_of_solvers =
  sizeof(list_of_all_solvers) / sizeof(std::string);

const unsigned int esbmc_num_solvers =
  sizeof(esbmc_solvers) / sizeof(esbmc_solver_config);

static smt_convt *create_solver(
  const std::string &&the_solver,
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api,
  const messaget &msg [[gnu::unused]])
{
  for(const auto &esbmc_solver : esbmc_solvers)
  {
    if(the_solver == esbmc_solver.name)
    {
      return esbmc_solver.create(
        options, ns, tuple_api, array_api, fp_api, msg);
    }
  }

  msg.error(fmt::format(
    "The {} solver has not been built into this version of ESBMC, sorry",
    the_solver));
  abort();
}

static const std::string pick_default_solver(const messaget &msg)
{
#ifdef BOOLECTOR
  msg.status("No solver specified; defaulting to Boolector");
  return "boolector";
#else
  // Pick whatever's first in the list.
  if(esbmc_num_solvers == 1)
  {
    msg.error(
      "No solver backends built into ESBMC; please either build "
      "some in, or explicitly configure the smtlib backend");
    abort();
  }
  else
  {
    msg.status(fmt::format(
      "No solver specified; defaulting to {}", esbmc_solvers[1].name));
    return esbmc_solvers[1].name;
  }
#endif
}

static smt_convt *pick_solver(
  const namespacet &ns,
  const optionst &options,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api,
  const messaget &msg)
{
  unsigned int i;
  std::string the_solver;

  for(i = 0; i < total_num_of_solvers; i++)
  {
    if(options.get_bool_option(list_of_all_solvers[i]))
    {
      if(the_solver != "")
      {
        msg.error("Please only specify one solver");
        abort();
      }

      the_solver = list_of_all_solvers[i];
    }
  }

  if(the_solver == "")
    the_solver = pick_default_solver(msg);

  return create_solver(
    std::move(the_solver), options, ns, tuple_api, array_api, fp_api, msg);
}

smt_convt *create_solver_factory1(
  const std::string &solver_name,
  const namespacet &ns,
  const optionst &options,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api,
  const messaget &msg)
{
  if(solver_name == "")
    // Pick one based on options.
    return pick_solver(ns, options, tuple_api, array_api, fp_api, msg);

  return create_solver(
    std::move(solver_name), options, ns, tuple_api, array_api, fp_api, msg);
}

smt_convt *create_solver_factory(
  const std::string &solver_name,
  const namespacet &ns,
  const optionst &options,
  const messaget &msg)
{
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  smt_convt *ctx = create_solver_factory1(
    solver_name, ns, options, &tuple_api, &array_api, &fp_api, msg);

  bool node_flat = options.get_bool_option("tuple-node-flattener");
  bool sym_flat = options.get_bool_option("tuple-sym-flattener");
  bool array_flat = options.get_bool_option("array-flattener");
  bool fp_to_bv = options.get_bool_option("fp2bv");

  // Pick a tuple flattener to use. If the solver has native support, and no
  // options were given, use that by default
  if(tuple_api != nullptr && !node_flat && !sym_flat)
    ctx->set_tuple_iface(tuple_api);
  // Use the node flattener if specified
  else if(node_flat)
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns, msg));
  // Use the symbol flattener if specified
  else if(sym_flat)
    ctx->set_tuple_iface(new smt_tuple_sym_flattener(ctx, ns, msg));
  // Default: node flattener
  else
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns, msg));

  // Pick an array flattener to use. Again, pick the solver native one by
  // default, or the one specified, or if none of the above then use the built
  // in arrays -> to BV flattener.
  if(array_api != nullptr && !array_flat)
    ctx->set_array_iface(array_api);
  else if(array_flat)
    ctx->set_array_iface(new array_convt(ctx));
  else
    ctx->set_array_iface(new array_convt(ctx));

  if(fp_api == nullptr || fp_to_bv)
    ctx->set_fp_conv(new fp_convt(ctx, msg));
  else
    ctx->set_fp_conv(fp_api);

  ctx->smt_post_init();
  return ctx;
}
