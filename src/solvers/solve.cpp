#include <solve.h>
#include <solvers/smt_conv.h>
#include <solvers/smt_solver.h>
#include <solver_config.h>

namespace
{
struct backendt
{
  /** Flag and --default-solver spelling. */
  const char *name;
  /** How the solver names itself in user-facing messages. */
  const char *display_name;
  bool built_in;
  /** Which camada solver to build; null for smtlib, whose factory branches. */
  camada_buildert build;
  /** Supports the Int/Real sorts that --ir and --ir-ieee emit. */
  bool int_real;
  /** Needs an external program the user must name, so it is never chosen
   *  implicitly. */
  bool needs_config;
};

/* Order is default priority: the first built-in backend that can be chosen
 * implicitly wins when the user names no solver. */
const backendt backends[] = {
  {"smtlib", "SMTLIB", ESBMC_ENABLE_smtlib, nullptr, true, true},
  {"bitwuzla",
   "bitwuzla",
   ESBMC_ENABLE_bitwuzla,
   create_esbmc_bitwuzla_solver,
   false,
   false},
  {"z3", "Z3", ESBMC_ENABLE_z3, create_esbmc_z3_solver, true, false},
  {"cvc5", "CVC5", ESBMC_ENABLE_cvc5, create_esbmc_cvc5_solver, true, false},
  {"mathsat",
   "MathSAT",
   ESBMC_ENABLE_mathsat,
   create_esbmc_mathsat_solver,
   true,
   false},
  {"yices",
   "Yices",
   ESBMC_ENABLE_yices,
   create_esbmc_yices_solver,
   true,
   false}};

const backendt *find_backend(const std::string &name)
{
  for (const backendt &b : backends)
    if (name == b.name)
      return &b;
  return nullptr;
}

[[noreturn]] void not_built_in(const std::string &name)
{
  log_error(
    "The {} solver has not been built into this version of ESBMC, sorry", name);
  abort();
}

/** The solver the user asked for, or "" if they did not. The flags are
 *  mutually exclusive. */
std::string user_choice(const optionst &options)
{
  std::string chosen;
  for (const backendt &b : backends)
    if (options.get_bool_option(b.name))
    {
      if (!chosen.empty())
      {
        log_error("Please only specify one solver");
        abort();
      }
      chosen = b.name;
    }

  return chosen.empty() ? options.get_option("default-solver") : chosen;
}

const backendt &pick_solver(const optionst &options)
{
  const bool int_encoding = options.get_bool_option("int-encoding");
  std::string name = user_choice(options);

  /* --ir and --ir-ieee both set int-encoding; keying off that rather than the
   * raw flag is what lets --ir-ieee auto-select too (issue #5179). */
  if (name.empty() && int_encoding)
  {
    for (const backendt &b : backends)
      if (b.built_in && !b.needs_config && b.int_real)
      {
        log_status(
          "Using integer/real arithmetic mode; defaulting to {}",
          b.display_name);
        name = b.name;
        break;
      }
    if (name.empty())
      log_warning(
        "No integer/real-capable solver built into this version of ESBMC; "
        "using default solver for integer/real mode");
  }

  if (name.empty())
    for (const backendt &b : backends)
      if (b.built_in && !b.needs_config)
      {
        log_status("No solver specified; defaulting to {}", b.display_name);
        name = b.name;
        break;
      }

  if (name.empty())
  {
    log_error(
      "No solver backends built into ESBMC; please either build "
      "some in, or explicitly configure the smtlib backend");
    abort();
  }

  const backendt *b = find_backend(name);
  if (!b || !b->built_in)
    not_built_in(name);

  /* Fail with a clear message instead of letting the backend abort() at
   * construction time (issue #5179). Reachable when a bit-vector-only solver
   * is forced via --default-solver together with --ir / --ir-ieee. */
  if (int_encoding && !b->int_real)
  {
    log_error(
      "Integer/real arithmetic (--ir / --ir-ieee) requires a solver that "
      "supports the Int/Real sorts (e.g. Z3); the '{}' backend is "
      "bit-vector-only. Re-run with an integer/real-capable solver, or build "
      "Z3 into ESBMC.",
      name);
    exit(1);
  }

  /* --smtlib-logic names the fragment the external solver accepts; a
   * quantifier-free bit-vector one cannot express the Int sorts --ir emits.
   * Reject the combination rather than hand the solver a script it will not
   * parse. */
  if (int_encoding && name == "smtlib")
  {
    const std::string logic = options.get_option("smtlib-logic");
    if (!logic.empty() && logic.find('I') == std::string::npos)
    {
      log_error(
        "Integer/real arithmetic (--ir / --ir-ieee) needs a logic with the "
        "Int sort, but --smtlib-logic is '{}'. Drop --smtlib-logic to let the "
        "encoding pick one, or name an integer-capable logic.",
        logic);
      exit(1);
    }
  }

  return *b;
}
} // namespace

void check_solver_availability(const optionst &options)
{
  /* Without an explicit choice, pick_solver() selects from whatever is built
   * in at solver-creation time. */
  const std::string name = user_choice(options);
  if (name.empty())
    return;

  const backendt *b = find_backend(name);
  if (!b || !b->built_in)
    not_built_in(name);
}

std::unique_ptr<smt_convt>
create_solver(const namespacet &ns, const optionst &options)
{
  /* The backend implements tuples, arrays and floating-point itself. Camada
     uses the solver's own theories where it has them and lowers otherwise, so
     there is no ESBMC-side flattener to install and nothing to select here.
     --fp2bv still forces FPEncoding::BV, because bit-blasting floats is a
     semantic choice a caller may need (see fp_encoding). */
  const backendt &backend = pick_solver(options);
  /* smtlib is the only backend whose construction branches; the rest differ
   * only in which camada solver they build. */
  std::unique_ptr<smt_solver_baset> ctx =
    backend.build ? create_linked_solver(options, ns, backend.build)
                  : create_new_smtlib_solver(options, ns);

  ctx->smt_post_init();
  return std::make_unique<smt_convt>(std::move(ctx));
}
