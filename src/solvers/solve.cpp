#include <solve.h>
#include <solver_config.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/fp/fp_conv.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_solver.h>
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
solver_creator create_new_bitwuzllob_solver;

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
  {"bitwuzla", create_new_bitwuzla_solver},
#endif
#ifdef BITWUZLLOB
  {"bitwuzllob", create_new_bitwuzllob_solver}
#endif
};

// Order encodes default priority: first compiled-in entry (excluding smtlib)
// is selected when no solver is explicitly requested.
static const std::string all_solvers[] = {
  "smtlib",
  "bitwuzllob",
  "bitwuzla",
  "boolector",
  "z3",
  "minisat",
  "cvc4",
  "cvc5",
  "mathsat",
  "yices"};

static std::string pick_default_solver()
{
  for (const std::string &name : all_solvers)
  {
    // smtlib and bitwuzllob depend on external programs the user must
    // configure, so they are never picked implicitly.
    if (name == "smtlib" || name == "bitwuzllob" || !esbmc_solvers.count(name))
      continue;
    log_status("No solver specified; defaulting to {}", name);
    return name;
  }
  log_error(
    "No solver backends built into ESBMC; please either build "
    "some in, or explicitly configure the smtlib backend");
  abort();
}

// Determine the solver the user explicitly asked for, returning "" if none.
// Aborts if the user requested more than one solver flag simultaneously.
static std::string resolve_user_solver_choice(const optionst &options)
{
  std::string solver_name;
  for (const std::string &name : all_solvers)
    if (options.get_bool_option(name))
    {
      if (!solver_name.empty())
      {
        log_error("Please only specify one solver");
        abort();
      }
      solver_name = name;
    }

  if (solver_name.empty())
    solver_name = options.get_option("default-solver");

  return solver_name;
}

void check_solver_availability(const optionst &options)
{
  std::string solver_name = resolve_user_solver_choice(options);
  // No explicit choice — pick_default_solver() will choose from what's built
  // in when the solver is actually needed.
  if (solver_name.empty())
    return;
  if (esbmc_solvers.count(solver_name))
    return;
  log_error(
    "The {} solver has not been built into this version of ESBMC, sorry",
    solver_name);
  abort();
}

static solver_creator &
pick_solver(std::string &solver_name, const optionst &options)
{
  if (solver_name == "")
    solver_name = resolve_user_solver_choice(options);

  // --ir and --ir-ieee both request integer/real arithmetic (both set the
  // "int-encoding" option). Default to Z3, which supports the Int/Real sorts,
  // when no solver was chosen. Keying off "int-encoding" rather than the raw
  // "ir" flag is what lets --ir-ieee auto-select too (issue #5179).
  if (solver_name == "" && options.get_bool_option("int-encoding"))
  {
#ifdef Z3
    if (esbmc_solvers.count("z3"))
    {
      log_status("Using integer/real arithmetic mode; defaulting to Z3");
      solver_name = "z3";
    }
    else
    {
      log_warning(
        "Z3 not available for integer/real arithmetic mode; using default "
        "solver");
    }
#else
    log_warning(
      "Z3 not built into this version of ESBMC; using default solver for "
      "integer/real mode");
#endif
  }
  if (solver_name == "")
    solver_name = pick_default_solver();

  // Integer/real encoding is incompatible with bit-vector-only backends
  // (Bitwuzla, Boolector). Fail with a clear message and a clean exit instead
  // of letting the backend abort() at construction time (issue #5179). This is
  // reachable when Z3 is not built in, or when a bit-vector-only solver is
  // forced via --default-solver together with --ir / --ir-ieee.
  if (
    options.get_bool_option("int-encoding") &&
    (solver_name == "bitwuzla" || solver_name == "bitwuzllob" ||
     solver_name == "boolector"))
  {
    log_error(
      "Integer/real arithmetic (--ir / --ir-ieee) requires a solver that "
      "supports the Int/Real sorts (e.g. Z3); the '{}' backend is "
      "bit-vector-only. Re-run with an integer/real-capable solver, or build "
      "Z3 into ESBMC.",
      solver_name);
    exit(1);
  }

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
  smt_solver_baset *ctx = factory(options, ns, &tuple_api, &array_api, &fp_api);

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
  return new smt_convt(std::unique_ptr<smt_solver_baset>(ctx));
}
