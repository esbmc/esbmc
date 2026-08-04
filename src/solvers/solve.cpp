#include <solve.h>
#include <solver_config.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/fp/fp_conv.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple_node.h>

#include <unordered_map>

solver_creator create_new_smtlib_solver;
solver_creator create_new_z3_solver;
solver_creator create_new_minisat_solver;
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
};

// Order encodes default priority: the first compiled-in entry that is not
// smtlib is selected when no solver is explicitly requested (smtlib needs an
// external program the user must name; see pick_default_solver).
static const std::string all_solvers[] = {
  "smtlib",
  "bitwuzla",
  "z3",
  "minisat",
  "cvc5",
  "mathsat",
  "yices"};

static std::string pick_default_solver()
{
  for (const std::string &name : all_solvers)
  {
    // smtlib depends on an external program the user must configure, so it is
    // never picked implicitly.
    if (name == "smtlib" || !esbmc_solvers.count(name))
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

  // Integer/real encoding is incompatible with bit-vector-only backends.
  // Fail with a clear message and a clean exit instead
  // of letting the backend abort() at construction time (issue #5179). This is
  // reachable when Z3 is not built in, or when a bit-vector-only solver is
  // forced via --default-solver together with --ir / --ir-ieee.
  if (options.get_bool_option("int-encoding") && solver_name == "bitwuzla")
  {
    log_error(
      "Integer/real arithmetic (--ir / --ir-ieee) requires a solver that "
      "supports the Int/Real sorts (e.g. Z3); the '{}' backend is "
      "bit-vector-only. Re-run with an integer/real-capable solver, or build "
      "Z3 into ESBMC.",
      solver_name);
    exit(1);
  }

  /* --smtlib-logic names the fragment the external solver accepts; a
   * quantifier-free bit-vector one cannot express the Int sorts --ir emits.
   * Reject the combination rather than hand the solver a script it will not
   * parse. */
  if (options.get_bool_option("int-encoding") && solver_name == "smtlib")
  {
    const std::string logic = options.get_option("smtlib-logic");
    if (!logic.empty() && logic.find("I") == std::string::npos)
    {
      log_error(
        "Integer/real arithmetic (--ir / --ir-ieee) needs a logic with the "
        "Int sort, but --smtlib-logic is '{}'. Drop --smtlib-logic to let the "
        "encoding pick one, or name an integer-capable logic.",
        logic);
      exit(1);
    }
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

  /* Use the solver's native tuples when it has them and nothing was asked
     for; otherwise flatten to per-field symbols. */
  if (tuple_api != nullptr && !node_flat)
    ctx->set_tuple_iface(tuple_api);
  else
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));

  /* --array-flattener is honoured by the backend: camada's Ackermann encoding
     keeps arrays out of the solver's array theory, so the interface is served
     either way and our own flattener is only needed by a backend that offers
     no arrays at all. */
  if (array_api != nullptr)
    ctx->set_array_iface(array_api);
  else
    ctx->set_array_iface(new array_convt(ctx));

  /* Every backend serves the FP interface itself: camada encodes
     floating-point natively or bit-blasts it (FPEncoding::BV, selected by
     --fp2bv), so ESBMC's software lowering is never installed. */
  assert(fp_api != nullptr);
  ctx->set_fp_conv(fp_api);

  ctx->smt_post_init();
  return new smt_convt(std::unique_ptr<smt_solver_baset>(ctx));
}
