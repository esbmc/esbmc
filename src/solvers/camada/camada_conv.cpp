#include <solvers/solve.h>
#include <solvers/smt/smt_array.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/camada/oneshot_options.h>
#include <solvers/smt/external_process_died.h>
#include <util/base/filesystem.h>

#include <util/arith/ieee_float.h>
#include <util/arith/mp_arith.h>

#include <camada/camada.h>
#include <camada/camadafeatures.h>
#include <camada/smtlibsolver.h>
#if CAMADA_HAVE_BITWUZLA
#  include <camada/bitwuzlasolver.h>
#endif
#if CAMADA_HAVE_MATHSAT
#  include <camada/mathsatsolver.h>
#endif
#if CAMADA_HAVE_YICES
#  include <camada/yicessolver.h>
#endif
#if CAMADA_HAVE_Z3
#  include <camada/z3solver.h>
#endif

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
enum class camada_backendt
{
  z3,
  cvc5,
  mathsat,
  yices,
  bitwuzla,
  smtlib
};

bool backend_supports_tuples(camada_backendt backend)
{
  switch (backend)
  {
  case camada_backendt::z3:
  case camada_backendt::cvc5:
  // TupleEncoding::Native emits SMT-LIB datatype tuples on the wire.
  case camada_backendt::smtlib:
    return true;
  case camada_backendt::mathsat:
  case camada_backendt::yices:
  case camada_backendt::bitwuzla:
    return false;
  }

  std::unreachable();
}

using camada_sort = solver_smt_sort<camada::SMTSortRef>;
using camada_expr = solver_smt_ast<camada::SMTExprRef>;

/* A model value that could not be read. `oneshot_name` is set when the model
 * comes from an external one-shot model solver: there, a failed read means
 * that process can no longer answer, and returning a default would invent
 * counterexample values, so report the dead process instead. Linked solvers
 * keep the historical warn-and-default behaviour. */
template <typename T>
static std::optional<T> unwrap_model_result(
  const camada::SMTResult<T> &result,
  std::string_view what,
  const char *oneshot_name = nullptr)
{
  if (result)
    return result.value();

  if (oneshot_name)
    throw external_process_died(
      std::string(oneshot_name) +
      ": the local model solver stopped answering "
      "while the counterexample was being read "
      "out (" +
      result.error().Message + ")");

  log_warning("Failed to extract {}: {}", what, result.error().Message);
  return std::nullopt;
}

class camada_tuple_ast : public camada_expr
{
public:
  using camada_expr::camada_expr;

  smt_astt update(
    smt_solver_baset *ctx,
    smt_astt value,
    unsigned int idx,
    const expr2tc &idx_expr) const override;

  smt_astt project(smt_solver_baset *ctx, unsigned int elem) const override;
};

#if CAMADA_HAVE_Z3
static void z3_error_handler(Z3_context c, Z3_error_code e)
{
  log_error("Z3 error {} encountered", Z3_get_error_msg(c, e));
  abort();
}

class esbmc_z3_solver : public camada::Z3Solver
{
public:
  explicit esbmc_z3_solver(z3::context c) : camada::Z3Solver(std::move(c))
  {
    setSolver(make_solver(camada::Z3Solver::context()));
  }

  void configure()
  {
    z3::params p(camada::Z3Solver::context());
    p.set("relevancy", 0U);
    p.set("model", true);
    p.set("proof", false);
    camada::Z3Solver::solver().set(p);
    Z3_set_ast_print_mode(
      camada::Z3Solver::context(), Z3_PRINT_SMTLIB2_COMPLIANT);
    Z3_set_error_handler(camada::Z3Solver::context(), z3_error_handler);
  }

  z3::context &context()
  {
    return camada::Z3Solver::context();
  }

private:
  // Pre-camada the Z3 backend built its solver from the same simplify ->
  // solve-eqs -> simplify -> smt tactic pipeline. The plain z3::solver(c)
  // camada used by default skips that preprocessing; restore the pipeline so
  // VC encoding mirrors the old behaviour (notably the simplification before
  // equality propagation, then a second simplify pass after).
  static z3::solver make_solver(z3::context &c)
  {
    return (z3::tactic(c, "simplify") & z3::tactic(c, "solve-eqs") &
            z3::tactic(c, "simplify") & z3::tactic(c, "smt"))
      .mk_solver();
  }
};
#endif

#if CAMADA_HAVE_YICES
class esbmc_yices_solver : public camada::YicesSolver
{
public:
  explicit esbmc_yices_solver(const char *logic, bool enable_push_pop)
    : logic(logic), enable_push_pop(enable_push_pop)
  {
    destroy_and_recreate();
  }

private:
  void destroy_and_recreate()
  {
    if (enable_push_pop)
      recreateContextWithConfig(logic, configure_push_pop);
    else
      recreateContext(logic);
  }

  static void configure_push_pop(ctx_config_t *config)
  {
    yices_set_config(config, "mode", "push-pop");
  }

  const char *logic;
  bool enable_push_pop;
};
#endif

#if CAMADA_HAVE_MATHSAT
class esbmc_mathsat_solver : public camada::MathSATSolver
{
public:
  explicit esbmc_mathsat_solver(const msat_config &config)
    : camada::MathSATSolver(config)
  {
  }
};
#endif

#if CAMADA_HAVE_BITWUZLA
class esbmc_bitwuzla_solver : public camada::BitwuzlaSolver
{
public:
  esbmc_bitwuzla_solver() = default;

  // Bitwuzla cannot decide equality over native constant arrays
  // ("Equality over constant arrays not fully supported yet" -> the solve
  // aborts). Opt into camada's lazy constant-array lowering, the same way
  // its Yices backend does, so the equalities ESBMC emits over list/dict
  // arrays are dischargeable.
  bool nativeConstArraySupport() const override
  {
    return false;
  }
};
#endif

camada::RM to_camada_rm(ieee_floatt::rounding_modet rm)
{
  switch (rm)
  {
  case ieee_floatt::UNKNOWN:
  case ieee_floatt::NONDETERMINISTIC:
    break;
  case ieee_floatt::ROUND_TO_EVEN:
    return camada::RM::ROUND_TO_EVEN;
  case ieee_floatt::ROUND_TO_AWAY:
    return camada::RM::ROUND_TO_AWAY;
  case ieee_floatt::ROUND_TO_PLUS_INF:
    return camada::RM::ROUND_TO_PLUS_INF;
  case ieee_floatt::ROUND_TO_MINUS_INF:
    return camada::RM::ROUND_TO_MINUS_INF;
  case ieee_floatt::ROUND_TO_ZERO:
    return camada::RM::ROUND_TO_ZERO;
  }

  log_error("Unsupported IEEE rounding mode {}", static_cast<int>(rm));
  abort();
}

[[noreturn]] void unsupported(const char *feature)
{
  log_error("Camada backend does not support {}", feature);
  abort();
}

std::string wrap_smtlib_dump(std::string smt_formula)
{
  std::replace(smt_formula.begin(), smt_formula.end(), '\\', '_');

  std::ostringstream dest;
  dest << "(set-info :smt-lib-version 2.6)\n";
  dest << "(set-option :produce-models true)\n";
  dest << "; Asserts from ESBMC starts\n";
  dest << smt_formula;
  if (!smt_formula.empty() && smt_formula.back() != '\n')
    dest << '\n';
  dest << "; Asserts from ESBMC ends\n";
  dest << "(get-model)\n";
  dest << "(exit)\n";
  return dest.str();
}

std::string pick_logic(const optionst &options, bool native_fp)
{
  const bool has_quantifiers = options.get_bool_option("has-quantifiers");

  if (options.get_bool_option("int-encoding"))
    return has_quantifiers ? "AUFLIRA" : "QF_AUFLIRA";

  if (options.get_bool_option("floatbv") || !native_fp)
    return has_quantifiers ? "AUFBV" : "QF_AUFBV";

  return has_quantifiers ? "AUFBVFP" : "QF_AUFBVFP";
}

/* The SMT-LIB backend pipes to an interactive solver
 * (--smtlib-solver-prog), writes the script out (--output, "-" for stdout),
 * or both. With neither, camada's write-only mode still needs a sink, so
 * default to stdout as the pre-camada backend's file_emitter did. */
camada::SMTSolverRef create_esbmc_smtlib_solver(const optionst &options)
{
  const std::string prog = options.get_option("smtlib-solver-prog");
  const std::string out = options.get_option("output");
  /* The SMT-LIB script goes to an arbitrary external solver, so name the
   * concrete fragment ESBMC's encoding produces rather than letting camada
   * negotiate `ALL` -- several regression tests grep this line, and a child
   * that only accepts concrete logics would otherwise get a retry it does
   * not need. native_fp is false: nothing here promises FP theory support. */
  const std::string logic = pick_logic(options, false);

  if (prog.empty())
    return std::make_unique<camada::SMTLIBSolver>(
      out.empty() ? "-" : out, camada::TupleEncoding::Native, logic);

  /* Camada spawns the child with execvp and no shell, so the command arrives
   * as an argv. Split on whitespace: enough for the documented "solver
   * executable and its flags" (e.g. "z3 -in"), and it keeps the child out of
   * a shell's reach. Quoting and metacharacters are deliberately NOT
   * interpreted -- the pre-camada backend ran the string through $SHELL -c,
   * which made this option an arbitrary-command sink. */
  if (prog.find_first_of("'\"\\$`|&;<>()*?") != std::string::npos)
  {
    log_error(
      "--smtlib-solver-prog is executed directly (execvp), not through a "
      "shell: '{}' contains shell metacharacters or quotes, which are not "
      "interpreted. Pass the executable and its flags as plain "
      "whitespace-separated words (e.g. \"z3 -in\").",
      prog);
    abort();
  }

  std::vector<std::string> argv;
  for (size_t i = 0; i < prog.size();)
  {
    size_t b = prog.find_first_not_of(" \t", i);
    if (b == std::string::npos)
      break;
    size_t e = prog.find_first_of(" \t", b);
    if (e == std::string::npos)
      e = prog.size();
    argv.emplace_back(prog, b, e - b);
    i = e;
  }

  /* Only mirror the script to a file when one was actually requested:
   * defaulting to stdout here would dump the whole formula on every piped
   * run. */
  if (out.empty())
    return std::make_unique<camada::SMTLIBSolver>(
      camada::SMTLIBProcessTag{}, argv, camada::TupleEncoding::Native, logic);

  return std::make_unique<camada::SMTLIBSolver>(
    camada::SMTLIBProcessTag{},
    argv,
    out,
    camada::TupleEncoding::Native,
    logic);
}

/* Re-scan a one-shot run's captured output with camada's own strict scanner
 * (last verdict wins, as in the live scan) so a solver that decided nothing
 * can be told apart from one that emitted no verdict at all. Substring
 * matching would misread a log line mentioning "unsat core". */
std::optional<camada::checkResult> tail_verdict(const std::string &tail)
{
  std::optional<camada::checkResult> found;
  size_t pos = 0;
  while (pos <= tail.size())
  {
    const size_t nl = tail.find('\n', pos);
    const size_t end = nl == std::string::npos ? tail.size() : nl;
    if (auto v = camada::parseOneShotVerdictLine(tail.substr(pos, end - pos)))
      found = v;
    if (nl == std::string::npos)
      break;
    pos = nl + 1;
  }
  return found;
}

/* Shared construction for the one-shot subprocess backends: the script goes
 * to formula_path, an external program is run on it once (%f -> the path),
 * and an optional local interactive solver is fed the same script in parallel
 * to serve get-value queries after a sat verdict. */
camada::SMTSolverRef create_esbmc_oneshot_solver(
  const optionst &options,
  const char *name,
  const std::string &formula_path,
  const std::string &prog,
  const std::string &logic)
{
  /* --smt-formula-only dumps the script and returns before dec_solve() (see
   * bmc.cpp), so no external solver ever runs. Build a write-only solver:
   * its check() emits the trailing (check-sat) and answers UNKNOWN without
   * spawning anything, which is exactly what the dump path needs. Camada's
   * one-shot check() fuses emitting and running, so asking for one-shot mode
   * here would launch the solver from the dump. */
  if (options.get_bool_option("smt-formula-only"))
    return std::make_unique<camada::SMTLIBSolver>(
      formula_path, camada::TupleEncoding::Native, logic);

  std::vector<std::string> model_argv;
  const std::string model = oneshot_options::model_prog(options, name);
  for (size_t i = 0; i < model.size();)
  {
    size_t b = model.find_first_not_of(" \t", i);
    if (b == std::string::npos)
      break;
    size_t e = model.find_first_of(" \t", b);
    if (e == std::string::npos)
      e = model.size();
    model_argv.emplace_back(model, b, e - b);
    i = e;
  }

  /* Register the child's process group so the timeout and signal handlers
   * tear down the whole solver subtree: they finish with _exit(), which runs
   * neither destructors nor atexit handlers, so camada's own cleanup cannot
   * cover those paths (an orphaned mpirun job would keep running). */
  return std::make_unique<camada::SMTLIBSolver>(
    camada::SMTLIBOneShotTag{},
    formula_path,
    prog,
    model_argv,
    [](long pgid) { file_operations::register_pgroup_for_cleanup(pgid); },
    camada::TupleEncoding::Native,
    logic);
}

camada::SMTSolverRef create_esbmc_z3_solver(const optionst &options)
{
#if CAMADA_HAVE_Z3
  std::string z3_file = options.get_option("z3-debug-dump-file");
  const bool z3_debug = options.get_bool_option("z3-debug");
  const bool smtlib2_compliant = options.get_bool_option("smt-formula-only") ||
                                 options.get_bool_option("smt-formula-too");

  // camada v0.11 reverted Z3Solver to take z3::context by value (with a move).
  // Build the configured context once and pass it through; the cfg branch only
  // fires for the z3-debug / smtlib-compliant modes.
  auto make_context = [&] {
    if (!z3_debug && !smtlib2_compliant)
      return z3::context();

    z3::config cfg;
    if (z3_debug)
    {
      Z3_open_log(z3_file.empty() ? "z3.log" : z3_file.c_str());
      cfg.set("stats", "true");
      cfg.set("type_check", "true");
      cfg.set("well_sorted_check", "true");
    }

    if (z3_debug || smtlib2_compliant)
      cfg.set("smtlib2_compliant", "true");

    return z3::context(cfg);
  };

  auto solver = std::make_unique<esbmc_z3_solver>(make_context());
  solver->configure();
  return solver;
#else
  unsupported("Z3 support in Camada");
#endif
}

camada::SMTSolverRef create_esbmc_mathsat_solver(const optionst &options)
{
#if CAMADA_HAVE_MATHSAT
  const std::string logic = pick_logic(options, true);
  msat_config config = msat_create_default_config(logic.c_str());
  msat_set_option(config, "model_generation", "true");
  msat_set_option(config, "preprocessor.toplevel_propagation", "true");
  msat_set_option(config, "preprocessor.simplification", "1");
  msat_set_option(config, "dpll.branching_random_frequency", "0.01");
  msat_set_option(
    config, "dpll.branching_random_invalidate_phase_cache", "true");
  msat_set_option(config, "dpll.restart_strategy", "3");
  msat_set_option(config, "dpll.glucose_var_activity", "true");
  msat_set_option(config, "dpll.glucose_learnt_minimization", "true");
  msat_set_option(config, "dpll.preprocessor.mode", "1");
  msat_set_option(config, "theory.bv.eager", "true");
  msat_set_option(config, "theory.bv.bit_blast_mode", "2");
  msat_set_option(config, "theory.bv.delay_propagated_eqs", "true");
  msat_set_option(config, "theory.la.enabled", "false");
  msat_set_option(config, "theory.fp.mode", "1");
  msat_set_option(config, "theory.fp.bit_blast_mode", "2");
  msat_set_option(config, "theory.fp.bv_combination_enabled", "true");
  msat_set_option(config, "theory.arr.enable_witness", "true");

  auto solver = std::make_unique<esbmc_mathsat_solver>(config);
  msat_destroy_config(config);
  return solver;
#else
  (void)options;
  unsupported("MathSAT support in Camada");
#endif
}

camada::SMTSolverRef create_esbmc_yices_solver(const optionst &options)
{
#if CAMADA_HAVE_YICES
  const std::string logic = pick_logic(options, false);
  return std::make_unique<esbmc_yices_solver>(
    logic.c_str(), options.get_bool_option("smt-during-symex"));
#else
  (void)options;
  unsupported("Yices support in Camada");
#endif
}

class camada_convt : public smt_solver_baset,
                     public tuple_iface,
                     public array_iface,
                     public fp_convt
{
  friend class camada_tuple_ast;

public:
  explicit camada_convt(
    const namespacet &ns,
    const optionst &options,
    camada_backendt backend,
    bool oneshot = false)
    : smt_solver_baset(ns, options),
      array_iface(true, true),
      fp_convt(this),
      backend(backend),
      oneshot(oneshot)
  {
    switch (backend)
    {
    case camada_backendt::z3:
      solver = create_esbmc_z3_solver(options);
      break;
    case camada_backendt::cvc5:
      solver = camada::createCVC5Solver();
      break;
    case camada_backendt::mathsat:
      solver = create_esbmc_mathsat_solver(options);
      break;
    case camada_backendt::yices:
      solver = create_esbmc_yices_solver(options);
      break;
    case camada_backendt::bitwuzla:
#if CAMADA_HAVE_BITWUZLA
      solver = std::make_unique<esbmc_bitwuzla_solver>();
#else
      unsupported("Bitwuzla support in Camada");
#endif
      break;
    case camada_backendt::smtlib:
      if (!oneshot)
      {
        solver = create_esbmc_smtlib_solver(options);
        break;
      }
      oneshot_prog = options.get_option("smtlib-oneshot-prog");
      formula_path = oneshot_options::choose_formula_path(options, "smtlib");
      /* An explicit --smtlib-logic is the user telling us what the external
       * solver accepts; otherwise follow the encoding. */
      const std::string logic = options.get_option("smtlib-logic");
      solver = create_esbmc_oneshot_solver(
        options,
        "smtlib",
        formula_path,
        oneshot_prog,
        logic.empty() ? pick_logic(options, false) : logic);
      break;
    }
  }

  void push_ctx() override
  {
    smt_solver_baset::push_ctx();
    solver->push();
  }

  void pop_ctx() override
  {
    smt_solver_baset::pop_ctx();
    solver->pop();
  }

  smt_resultt dec_solve() override
  {
    if (oneshot)
      return oneshot_dec_solve();

    pre_solve();

    switch (solver->check())
    {
    case camada::checkResult::SAT:
      return P_SATISFIABLE;
    case camada::checkResult::UNSAT:
      return P_UNSATISFIABLE;
    case camada::checkResult::UNKNOWN:
      return P_ERROR;
    }
    std::unreachable();
  }

  /* The one-shot solver answers with a verdict and then exits, so it cannot
   * serve (get-value): a satisfiable formula needs the parallel local model
   * solver to build a counterexample. Camada reports both verdicts and the
   * run's diagnostics; the policy and the messages stay here. */
  smt_resultt oneshot_dec_solve()
  {
    if (solved)
    {
      log_error(
        "the {} backend supports a single (check-sat) query per run; "
        "incremental strategies are not supported",
        "smtlib");
      abort();
    }
    solved = true;

    pre_solve();

    auto *smtlib = static_cast<camada::SMTLIBSolver *>(solver.get());
    const camada::checkResult res = smtlib->check();

    if (res != camada::checkResult::SAT)
    {
      if (res != camada::checkResult::UNKNOWN)
        return P_UNSATISFIABLE;

      /* Camada answers UNKNOWN for three distinct outcomes; they are worth
       * different reports, and the exit status tells them apart. */
      const camada::OneShotDiagnostics &d = smtlib->oneShotDiagnostics();
      const bool signalled = d.ExitStatus.compare(0, 7, "signal ") == 0;

      if (signalled)
        /* A verdict from a solver that died on a signal (crash, OOM kill, a
         * wrapper tearing the job down) cannot be trusted; camada already
         * discarded it. Non-zero exit codes stay accepted -- SAT-competition
         * style solvers exit 10/20. */
        log_error(
          "{}: solver command \"{}\" died with {}; discarding its verdict",
          "smtlib",
          d.Command,
          d.ExitStatus);
      else if (tail_verdict(d.OutputTail) == camada::checkResult::UNKNOWN)
        /* The solver decided nothing and said so. */
        log_error("{}: solver returned unknown", "smtlib");
      else
        log_error(
          "{}: no sat/unsat verdict in the output of \"{}\" ({}); last output "
          "lines:\n{}",
          "smtlib",
          d.Command,
          d.ExitStatus,
          d.OutputTail);

      return P_ERROR;
    }

    /* Check for divergence first: camada drops a model solver that disagrees
     * (it has no model to serve), so this must be distinguished from one that
     * never started or died -- otherwise a wrong-answer bug in the one-shot
     * solver is reported as a missing model. */
    const std::optional<camada::checkResult> model =
      smtlib->oneShotModelVerdict();
    if (model && *model != camada::checkResult::SAT)
    {
      log_error(
        "{}: {} reported sat but the local model solver did not; refusing to "
        "build a counterexample from a diverging model",
        "smtlib",
        oneshot_prog);
      abort();
    }

    /* A solver that agreed and then went away is not "unavailable": the
     * verdict stands, and the readout paths report the death via
     * external_process_died when the counterexample is built. */
    if (
      !smtlib->oneShotModelSolverLive() &&
      smtlib->oneShotModelVerdict() != camada::checkResult::SAT)
    {
      if (options.get_bool_option("result-only"))
        return P_SATISFIABLE;
      if (options.get_option("smtlib-oneshot-model-prog").empty())
        log_error(
          "{}: formula is satisfiable, but building the counterexample "
          "requires a local interactive SMT-LIB2 solver; re-run with "
          "--smtlib-oneshot-model-prog <cmd> (e.g. \"z3 -in\") or with "
          "--result-only",
          "smtlib");
      else
        log_error(
          "{}: the local model solver is unavailable; cannot build a "
          "counterexample",
          "smtlib");
      return P_ERROR;
    }

    return P_SATISFIABLE;
  }

  void assert_ast(smt_astt a) override
  {
    solver->addConstraint(to_solver_smt_ast<camada_expr>(a)->a);
  }

  /* Non-null only when the model comes from an external one-shot model
   * solver, whose disappearance must be reported rather than defaulted. */
  const char *oneshot_label() const
  {
    return oneshot ? "smtlib" : nullptr;
  }

  tvt get_bool(smt_astt a) override
  {
    const auto value = expr(a);
    const auto kind = value->getKind();
    if (
      kind == camada::SMTExprKind::Forall ||
      kind == camada::SMTExprKind::Exists)
    {
      log_warning(
        "Skipping concrete model extraction for quantified boolean term");
      return tvt(tvt::TV_UNKNOWN);
    }

    std::string dump;
    value->dump(dump);
    if (
      dump.find("(forall ") != std::string::npos ||
      dump.find("(exists ") != std::string::npos)
    {
      log_warning(
        "Skipping concrete model extraction for boolean term containing "
        "quantifiers");
      return tvt(tvt::TV_UNKNOWN);
    }

    auto result = unwrap_model_result(
      solver->getBool(value), "boolean model value", oneshot_label());
    if (!result)
      return tvt(tvt::TV_UNKNOWN);

    return tvt(*result);
  }

  BigInt get_bv(smt_astt a, bool is_signed) override
  {
    const auto exp = to_solver_smt_ast<camada_expr>(a)->a;
    if (int_encoding)
    {
      if (exp->isRealSort())
      {
        auto result = unwrap_model_result(
          solver->getRational(exp), "rational model value", oneshot_label());
        if (!result)
          return BigInt(0);

        BigInt num = string2integer(result->first);
        BigInt den = string2integer(result->second);
        return num / den;
      }

      auto result = unwrap_model_result(
        solver->getInt(exp), "integer model value", oneshot_label());
      return result ? string2integer(*result) : BigInt(0);
    }

    auto result = unwrap_model_result(
      solver->getBVInBin(exp), "bit-vector model value", oneshot_label());
    return result ? binary2integer(*result, is_signed) : BigInt(0);
  }

  ieee_floatt get_fpbv(smt_astt a) override
  {
    auto model_result = unwrap_model_result(
      solver->getFPInBin(to_solver_smt_ast<camada_expr>(a)->a),
      "floating-point model value",
      oneshot_label());
    if (!model_result)
      return ieee_floatt(ieee_float_spect(
        a->sort->get_significand_width() - 1, a->sort->get_exponent_width()));

    std::string bits = *model_result;
    const auto ew = a->sort->get_exponent_width();
    const auto sw = a->sort->get_significand_width() - 1;
    ieee_floatt result(ieee_float_spect(sw, ew));
    result.unpack(binary2integer(bits, false));
    return result;
  }

  bool get_rational(smt_astt a, BigInt &numerator, BigInt &denominator) override
  {
    auto result = unwrap_model_result(
      solver->getRational(to_solver_smt_ast<camada_expr>(a)->a),
      "rational model value",
      oneshot_label());
    if (!result)
      return false;

    numerator = BigInt(result->first.c_str(), 10);
    denominator = BigInt(result->second.c_str(), 10);
    return true;
  }

  expr2tc get_array_elem(smt_astt array, uint64_t index, const type2tc &subtype)
    override
  {
    const auto *array_ast = to_solver_smt_ast<camada_expr>(array);
    const auto index_sort = array_ast->a->Sort->getIndexSort();
    auto idx = make_index_expr(index_sort, index);
    auto elem = solver->getArrayElement(array_ast->a, idx);
    auto elem_sort = convert_sort(subtype);
    return get_by_ast(subtype, new camada_expr(this, elem, elem_sort));
  }

  smt_astt mk_add(smt_astt a, smt_astt b) override
  {
    return wrap_binary(a, b, [this](const auto &lhs, const auto &rhs) {
      return lhs->isArithSort() ? solver->mkArithAdd(lhs, rhs)
                                : solver->mkBVAdd(lhs, rhs);
    });
  }

  smt_astt mk_bvadd(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVAdd(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_sub(smt_astt a, smt_astt b) override
  {
    return wrap_binary(a, b, [this](const auto &lhs, const auto &rhs) {
      return lhs->isArithSort() ? solver->mkArithSub(lhs, rhs)
                                : solver->mkBVSub(lhs, rhs);
    });
  }
  smt_astt mk_bvsub(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSub(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_mul(smt_astt a, smt_astt b) override
  {
    return wrap_binary(a, b, [this](const auto &lhs, const auto &rhs) {
      return lhs->isArithSort() ? solver->mkArithMul(lhs, rhs)
                                : solver->mkBVMul(lhs, rhs);
    });
  }
  smt_astt mk_bvmul(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVMul(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_mod(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkArithMod(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvsmod(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSRem(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvumod(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVURem(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_div(smt_astt a, smt_astt b) override
  {
    return wrap_binary(a, b, [this](const auto &lhs, const auto &rhs) {
      return lhs->isArithSort() ? solver->mkArithDiv(lhs, rhs)
                                : solver->mkBVUDiv(lhs, rhs);
    });
  }
  smt_astt mk_bvsdiv(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSDiv(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvudiv(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVUDiv(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_shl(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkArithShl(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvshl(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVShl(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvashr(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVAshr(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvlshr(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVLshr(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_neg(smt_astt a) override
  {
    auto ea = expr(a);
    return wrap(
      ea->isArithSort() ? solver->mkArithNeg(ea) : solver->mkBVNeg(ea),
      a->sort);
  }
  smt_astt mk_bvneg(smt_astt a) override
  {
    return wrap(solver->mkBVNeg(expr(a)), a->sort);
  }
  smt_astt mk_bvnot(smt_astt a) override
  {
    if (int_encoding)
      return int_bitwise_unary(
        a, [this](const camada::SMTExprRef &v) { return solver->mkBVNot(v); });
    return wrap(solver->mkBVNot(expr(a)), a->sort);
  }
  smt_astt mk_bvxor(smt_astt a, smt_astt b) override
  {
    if (int_encoding)
      return int_bitwise_binary(
        a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
          return solver->mkBVXor(l, r);
        });
    return wrap(solver->mkBVXor(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvor(smt_astt a, smt_astt b) override
  {
    if (int_encoding)
      return int_bitwise_binary(
        a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
          return solver->mkBVOr(l, r);
        });
    return wrap(solver->mkBVOr(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_bvand(smt_astt a, smt_astt b) override
  {
    if (int_encoding)
      return int_bitwise_binary(
        a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
          return solver->mkBVAnd(l, r);
        });
    return wrap(solver->mkBVAnd(expr(a), expr(b)), a->sort);
  }
  smt_astt mk_implies(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkImplies(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_xor(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkXor(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_or(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkOr(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_and(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkAnd(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_not(smt_astt a) override
  {
    return wrap(solver->mkNot(expr(a)), boolean_sort);
  }
  smt_astt mk_lt(smt_astt a, smt_astt b) override
  {
    return wrap_binary(
      a,
      b,
      [this](const auto &lhs, const auto &rhs) {
        return lhs->isArithSort() ? solver->mkArithLt(lhs, rhs)
                                  : solver->mkBVUlt(lhs, rhs);
      },
      boolean_sort);
  }
  smt_astt mk_bvult(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVUlt(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_bvslt(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSlt(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_gt(smt_astt a, smt_astt b) override
  {
    return wrap_binary(
      a,
      b,
      [this](const auto &lhs, const auto &rhs) {
        return lhs->isArithSort() ? solver->mkArithGt(lhs, rhs)
                                  : solver->mkBVUgt(lhs, rhs);
      },
      boolean_sort);
  }
  smt_astt mk_bvugt(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVUgt(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_bvsgt(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSgt(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_le(smt_astt a, smt_astt b) override
  {
    return wrap_binary(
      a,
      b,
      [this](const auto &lhs, const auto &rhs) {
        return lhs->isArithSort() ? solver->mkArithLe(lhs, rhs)
                                  : solver->mkBVUle(lhs, rhs);
      },
      boolean_sort);
  }
  smt_astt mk_bvule(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVUle(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_bvsle(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSle(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_ge(smt_astt a, smt_astt b) override
  {
    return wrap_binary(
      a,
      b,
      [this](const auto &lhs, const auto &rhs) {
        return lhs->isArithSort() ? solver->mkArithGe(lhs, rhs)
                                  : solver->mkBVUge(lhs, rhs);
      },
      boolean_sort);
  }
  smt_astt mk_bvuge(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVUge(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_bvsge(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkBVSge(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_eq(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkEqual(expr(a), expr(b)), boolean_sort);
  }
  smt_astt mk_neq(smt_astt a, smt_astt b) override
  {
    return wrap(solver->mkNot(solver->mkEqual(expr(a), expr(b))), boolean_sort);
  }
  smt_astt mk_store(smt_astt a, smt_astt b, smt_astt c) override
  {
    return wrap(solver->mkArrayStore(expr(a), expr(b), expr(c)), a->sort);
  }
  smt_astt mk_select(smt_astt a, smt_astt b) override
  {
    return wrap(
      solver->mkArraySelect(expr(a), expr(b)), a->sort->get_range_sort());
  }
  smt_astt mk_real2int(smt_astt a) override
  {
    return wrap(solver->mkReal2Int(expr(a)), mk_int_sort());
  }
  smt_astt mk_int2real(smt_astt a) override
  {
    return wrap(solver->mkInt2Real(expr(a)), mk_real_sort());
  }
  smt_astt mk_isint(smt_astt a) override
  {
    return wrap(solver->mkIsInt(expr(a)), boolean_sort);
  }

  smt_sortt mk_bool_sort() override
  {
    return new camada_sort(SMT_SORT_BOOL, solver->mkBoolSort(), 1);
  }

  smt_sortt mk_real_sort() override
  {
    return new camada_sort(SMT_SORT_REAL, solver->mkRealSort());
  }

  smt_sortt mk_int_sort() override
  {
    return new camada_sort(SMT_SORT_INT, solver->mkIntSort());
  }

  smt_sortt mk_bv_sort(std::size_t width) override
  {
    return new camada_sort(SMT_SORT_BV, solver->mkBVSort(width), width);
  }

  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override
  {
    auto cam_domain = to_solver_smt_sort<camada::SMTSortRef>(domain)->s;
    auto cam_range = to_solver_smt_sort<camada::SMTSortRef>(range)->s;
    auto sort = solver->mkArraySort(cam_domain, cam_range);
    return new camada_sort(
      SMT_SORT_ARRAY, sort, domain->get_data_width(), range);
  }

  smt_sortt mk_fbv_sort(std::size_t width) override
  {
    return new camada_sort(SMT_SORT_FIXEDBV, solver->mkBVSort(width), width);
  }

  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw) override
  {
    auto sort = solver->mkBVSort(ew + sw + 1);
    return new camada_sort(SMT_SORT_BVFP, sort, ew + sw + 1, sw + 1);
  }

  smt_sortt mk_bvfp_rm_sort() override
  {
    auto sort = solver->mkBVSort(3);
    return new camada_sort(SMT_SORT_BVFP_RM, sort, 3);
  }

  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw) override
  {
    auto sort = solver->mkFPSort(ew, sw, fp_encoding());
    return new camada_sort(SMT_SORT_FPBV, sort, ew + sw + 1, sw + 1);
  }

  smt_sortt mk_fpbv_rm_sort() override
  {
    return new camada_sort(
      SMT_SORT_FPBV_RM, solver->mkRMSort(fp_encoding()), 3);
  }

  smt_astt mk_smt_int(const BigInt &theint) override
  {
    return wrap(solver->mkInt(integer2string(theint, 10)), mk_int_sort());
  }

  smt_astt mk_smt_real(const std::string &str) override
  {
    return wrap(solver->mkReal(str), mk_real_sort());
  }

  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s) override
  {
    return wrap(
      solver->mkBVFromBin(
        integer2binary(theint, s->get_data_width()),
        to_solver_smt_sort<camada::SMTSortRef>(s)->s),
      s);
  }

  smt_astt mk_smt_fpbv(const ieee_floatt &thereal) override
  {
    std::string bits = integer2binary(thereal.pack(), thereal.spec.width());
    return wrap(
      solver->mkFPFromBin(bits, thereal.spec.e, fp_encoding()),
      mk_fpbv_sort(thereal.spec.e, thereal.spec.f));
  }

  smt_astt mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw) override
  {
    return wrap(
      solver->mkNaN(sgn, ew, sw, fp_encoding()), mk_fpbv_sort(ew, sw - 1));
  }

  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw) override
  {
    return wrap(
      solver->mkInf(sgn, ew, sw, fp_encoding()), mk_fpbv_sort(ew, sw - 1));
  }

  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm) override
  {
    auto sort = mk_fpbv_rm_sort();
    return wrap(solver->mkRM(to_camada_rm(rm), fp_encoding()), sort);
  }

  smt_astt
  mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm) override
  {
    return wrap(
      solver->mkFPFMA(expr(v1), expr(v2), expr(v3), expr(rm)), v1->sort);
  }

  smt_astt
  mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width) override
  {
    return wrap(solver->mkFPtoUBV(expr(from), width), mk_bv_sort(width));
  }

  smt_astt
  mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width) override
  {
    return wrap(solver->mkFPtoSBV(expr(from), width), mk_bv_sort(width));
  }

  smt_astt mk_smt_typecast_from_fpbv_to_fpbv(
    smt_astt from,
    smt_sortt to,
    smt_astt rm) override
  {
    return wrap(
      solver->mkFPtoFP(
        expr(from), to_solver_smt_sort<camada::SMTSortRef>(to)->s, expr(rm)),
      to);
  }

  smt_astt
  mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm) override
  {
    return wrap(
      solver->mkUBVtoFP(
        expr(from), to_solver_smt_sort<camada::SMTSortRef>(to)->s, expr(rm)),
      to);
  }

  smt_astt
  mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm) override
  {
    return wrap(
      solver->mkSBVtoFP(
        expr(from), to_solver_smt_sort<camada::SMTSortRef>(to)->s, expr(rm)),
      to);
  }

  smt_astt mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm) override
  {
    return wrap(solver->mkFPAdd(expr(lhs), expr(rhs), expr(rm)), lhs->sort);
  }
  smt_astt mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm) override
  {
    return wrap(solver->mkFPSub(expr(lhs), expr(rhs), expr(rm)), lhs->sort);
  }
  smt_astt mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm) override
  {
    return wrap(solver->mkFPMul(expr(lhs), expr(rhs), expr(rm)), lhs->sort);
  }
  smt_astt mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm) override
  {
    return wrap(solver->mkFPDiv(expr(lhs), expr(rhs), expr(rm)), lhs->sort);
  }
  smt_astt mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm) override
  {
    return wrap(solver->mkFPtoIntegral(expr(from), expr(rm)), from->sort);
  }
  smt_astt mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm) override
  {
    return wrap(solver->mkFPSqrt(expr(rd), expr(rm)), rd->sort);
  }

  smt_astt mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs) override
  {
    return wrap(solver->mkFPEqual(expr(lhs), expr(rhs)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs) override
  {
    return wrap(solver->mkFPGt(expr(lhs), expr(rhs)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs) override
  {
    return wrap(solver->mkFPLt(expr(lhs), expr(rhs)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs) override
  {
    return wrap(solver->mkFPGe(expr(lhs), expr(rhs)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs) override
  {
    return wrap(solver->mkFPLe(expr(lhs), expr(rhs)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_is_nan(smt_astt op) override
  {
    return wrap(solver->mkFPIsNaN(expr(op)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_is_inf(smt_astt op) override
  {
    return wrap(solver->mkFPIsInfinite(expr(op)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_is_normal(smt_astt op) override
  {
    return wrap(solver->mkFPIsNormal(expr(op)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_is_zero(smt_astt op) override
  {
    return wrap(solver->mkFPIsZero(expr(op)), boolean_sort);
  }
  smt_astt mk_smt_fpbv_is_negative(smt_astt op) override
  {
    return fp_sign_test(op, true);
  }
  smt_astt mk_smt_fpbv_is_positive(smt_astt op) override
  {
    return fp_sign_test(op, false);
  }
  smt_astt mk_smt_fpbv_abs(smt_astt op) override
  {
    return wrap(solver->mkFPAbs(expr(op)), op->sort);
  }
  smt_astt mk_smt_fpbv_neg(smt_astt op) override
  {
    return wrap(solver->mkFPNeg(expr(op)), op->sort);
  }

  smt_astt mk_from_bv_to_fp(smt_astt op, smt_sortt to) override
  {
    return wrap(
      solver->mkBVToIEEEFP(
        expr(op), to_solver_smt_sort<camada::SMTSortRef>(to)->s),
      to);
  }

  smt_astt mk_from_fp_to_bv(smt_astt op) override
  {
    auto to = mk_bvfp_sort(
      op->sort->get_exponent_width(), op->sort->get_significand_width() - 1);
    return wrap(solver->mkIEEEFPToBV(expr(op)), to);
  }

  smt_astt mk_smt_bool(bool val) override
  {
    return wrap(solver->mkBool(val), boolean_sort);
  }

  smt_astt
  mk_array_symbol(const std::string &name, smt_sortt sort, smt_sortt) override
  {
    return mk_smt_symbol(name, sort);
  }

  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override
  {
    return wrap(
      solver->mkSymbol(name, to_solver_smt_sort<camada::SMTSortRef>(s)->s), s);
  }

  smt_sortt mk_struct_sort(const type2tc &type) override
  {
    if (is_array_type(type))
    {
      const array_type2t &arrtype = to_array_type(type);
      smt_sortt subtypesort = convert_sort(arrtype.subtype);
      smt_sortt d =
        mk_int_bv_sort(make_array_domain_type(arrtype)->get_width());
      return mk_array_sort(d, subtypesort);
    }

    const std::vector<type2tc> &members = struct_union_members(type);
    std::vector<camada::SMTSortRef> field_sorts;
    field_sorts.reserve(members.size());
    for (const auto &member : members)
      field_sorts.push_back(
        to_solver_smt_sort<camada::SMTSortRef>(convert_sort(member))->s);

    return new camada_sort(
      SMT_SORT_STRUCT, solver->mkTupleSort(field_sorts), type);
  }

  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low) override
  {
    // If it's a floatbv, convert it to bv first so callers extracting bytes
    // out of structs/unions containing floats encode against the IEEE bit
    // pattern instead of triggering a sort mismatch.
    if (a->sort->id == SMT_SORT_FPBV)
      a = mk_from_fp_to_bv(a);
    return wrap(
      solver->mkBVExtract(high, low, expr(a)), mk_bv_sort(high - low + 1));
  }

  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override
  {
    return wrap(
      solver->mkBVSignExt(topwidth, expr(a)),
      mk_bv_sort(a->sort->get_data_width() + topwidth));
  }

  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override
  {
    return wrap(
      solver->mkBVZeroExt(topwidth, expr(a)),
      mk_bv_sort(a->sort->get_data_width() + topwidth));
  }

  smt_astt mk_concat(smt_astt a, smt_astt b) override
  {
    return wrap(
      solver->mkBVConcat(expr(a), expr(b)),
      mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width()));
  }

  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override
  {
    return wrap(solver->mkIte(expr(cond), expr(t), expr(f)), t->sort);
  }

  smt_astt tuple_create(const expr2tc &structdef) override
  {
    const constant_struct2t &strct = to_constant_struct2t(structdef);
    std::vector<camada::SMTExprRef> fields;
    fields.reserve(strct.datatype_members.size());
    for (const auto &member : strct.datatype_members)
      fields.push_back(expr(convert_ast(member)));

    return wrap(solver->mkTuple(fields), mk_struct_sort(structdef->type));
  }

  smt_astt tuple_fresh(const smt_sort *s, std::string name = "") override
  {
    if (name.empty())
      name = mk_fresh_name("camada_convt::tuple_fresh");
    return wrap(
      solver->mkSymbol(name, to_solver_smt_sort<camada::SMTSortRef>(s)->s), s);
  }

  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *input_args,
    bool const_array,
    const smt_sort *domain) override
  {
    const array_type2t &arrtype = to_array_type(array_type);
    smt_sortt elem_sort = mk_struct_sort(arrtype.subtype);
    smt_sortt array_sort = mk_array_sort(domain, elem_sort);

    if (const_array)
    {
      return wrap(
        solver->mkArrayConst(
          to_solver_smt_sort<camada::SMTSortRef>(domain)->s, expr(*input_args)),
        array_sort);
    }

    assert(!is_nil_expr(arrtype.array_size));
    assert(is_constant_int2t(arrtype.array_size));

    auto result = solver->mkSymbol(
      mk_fresh_name("camada_convt::tuple_array_create"),
      to_solver_smt_sort<camada::SMTSortRef>(array_sort)->s);
    auto domain_sort = to_solver_smt_sort<camada::SMTSortRef>(domain)->s;

    for (std::size_t i = 0;
         i < to_constant_int2t(arrtype.array_size).as_ulong();
         ++i)
    {
      result = solver->mkArrayStore(
        result, make_index_expr(domain_sort, i), expr(input_args[i]));
    }

    return wrap(result, array_sort);
  }

  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override
  {
    if (name == "NULL")
      return null_ptr_ast;

    if (name == "INVALID")
      return invalid_ptr_ast;

    return mk_smt_symbol(name, s);
  }

  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override
  {
    const symbol2t &sym = to_symbol2t(expr);
    return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
  }

  smt_astt
  tuple_array_of(const expr2tc &init_value, unsigned long domain_width) override
  {
    return convert_array_of(convert_ast(init_value), domain_width);
  }

  expr2tc tuple_get(const expr2tc &expr) override
  {
    return tuple_get(expr->type, convert_ast(expr));
  }

  expr2tc tuple_get(const type2tc &type, smt_astt sym) override
  {
    // Pointer types lower to pointer_struct in SMT; struct_union_members(type)
    // would throw on the raw pointer type, so dispatch the pointer case first.
    const std::vector<type2tc> &members =
      struct_union_members(is_pointer_type(type) ? pointer_struct : type);

    if (is_pointer_type(type))
    {
      smt_astt object =
        wrap(solver->mkTupleSelect(expr(sym), 0), convert_sort(members[0]));
      smt_astt offset =
        wrap(solver->mkTupleSelect(expr(sym), 1), convert_sort(members[1]));

      unsigned int num =
        get_bv(object, is_signedbv_type(members[0])).to_uint64();
      unsigned int offs =
        get_bv(offset, is_signedbv_type(members[1])).to_uint64();
      pointer_logict::pointert p(num, BigInt(offs));
      return pointer_logic.back().pointer_expr(p, type);
    }

    std::vector<expr2tc> outmem;
    outmem.reserve(members.size());
    for (std::size_t i = 0; i < members.size(); ++i)
    {
      outmem.push_back(get_by_ast(
        members[i],
        wrap(solver->mkTupleSelect(expr(sym), i), convert_sort(members[i]))));
    }

    return constant_struct2tc(type, std::move(outmem));
  }

  expr2tc tuple_get_array_elem(
    smt_astt array,
    uint64_t index,
    const type2tc &subtype) override
  {
    return get_array_elem(array, index, get_flattened_array_subtype(subtype));
  }

  smt_astt
  convert_array_of(smt_astt init_val, unsigned long domain_width) override
  {
    auto idx_sort = int_encoding
                      ? solver->mkIntSort()
                      : solver->mkBVSort(domain_width == 0 ? 1 : domain_width);
    auto value = solver->mkArrayConst(idx_sort, expr(init_val));
    return wrap(value, from_camada_sort(value->Sort));
  }

  ~camada_convt() override
  {
    /* A temporary formula file is ours to remove; a --output path is the
     * user's. The signal/timeout paths skip this via _exit(), which is why
     * choose_formula_path() also registers the temp for cleanup there. */
    if (oneshot && oneshot_options::uses_temp_formula(options))
      remove(formula_path.c_str());
  }

  const std::string solver_text() override
  {
    if (oneshot)
      return "one-shot '" + oneshot_prog + "'";
    return solver->getSolverNameAndVersion();
  }

  std::string dump_smt() override
  {
    /* Camada's SMT-LIB backend streams the script to its sink as it is built
     * rather than buffering it, so there is nothing to hand back. Complete the
     * script with the (check-sat) that --smt-formula-only never reaches via
     * dec_solve() -- in write-only mode check() just emits it and answers
     * UNKNOWN -- then return empty so bmc.cpp does not reopen the same path
     * and overwrite what camada wrote (issue #6059). */
    if (backend == camada_backendt::smtlib && !oneshot)
    {
      solver->check();
      const std::string path = options.get_option("output");
      if (path.empty() || path == "-")
        log_status("SMT formula written to standard output");
      else
        log_status("SMT formula written to output file {}", path);
      return "";
    }

    if (oneshot)
    {
      /* Under --smt-formula-only no solve follows, so close the script here;
       * the solver is write-only in that mode, so check() only emits the
       * trailing (check-sat). Under --smt-formula-too our dec_solve() emits
       * it instead: a second one would hand the external solver a script
       * containing two. Either way camada already wrote the formula, so
       * return empty to keep bmc.cpp from overwriting it (issue #6059). */
      if (options.get_bool_option("smt-formula-only"))
      {
        solver->check();
        if (formula_path == "-")
          log_status("SMT formula written to standard output");
        else
          log_status("SMT formula written to output file {}", formula_path);
        return "";
      }

      log_status("SMT formula written to {}", formula_path);
      return "";
    }

    std::string smt_formula;
    solver->dump(smt_formula);
    return wrap_smtlib_dump(std::move(smt_formula));
  }

  void print_model() override
  {
    solver->dumpModel();
  }

  smt_astt mk_quantifier(
    bool is_forall,
    std::vector<smt_astt> lhs,
    smt_astt rhs) override
  {
    std::vector<camada::SMTExprRef> vars;
    vars.reserve(lhs.size());
    for (const auto &var : lhs)
      vars.push_back(expr(var));

    auto q = is_forall ? solver->mkForall(vars, expr(rhs))
                       : solver->mkExists(vars, expr(rhs));
    return wrap(q, boolean_sort);
  }

private:
  std::unique_ptr<camada::SMTSolver> solver;
  const camada_backendt backend;
  /* Set when the SMT-LIB backend drives an external one-shot program. */
  const bool oneshot = false;
  std::string oneshot_prog;
  std::string formula_path;
  bool solved = false;

  static camada::SMTExprRef expr(smt_astt a)
  {
    return to_solver_smt_ast<camada_expr>(a)->a;
  }

  /* --fp2bv asks for floating-point encoded as bit-vectors, which is exactly
   * camada's FPEncoding::BV; let it bit-blast rather than swapping in ESBMC's
   * own software lowering. Keyed off fp2bv, not floatbv: the latter is set by
   * default for anything that is not --fixedbv, so reading it made every run
   * bit-blast and native FP unreachable. */
  camada::FPEncoding fp_encoding() const
  {
    return options.get_bool_option("fp2bv") ? camada::FPEncoding::BV
                                            : camada::FPEncoding::Native;
  }

  /* SMT-LIB's Int/Real theory has no bitwise operations, but ESBMC still
   * routes C's &/|/^/~ through the mk_bv* entry points in integer mode (see
   * the bitand_id case in smt_solver.cpp). Bridge by converting to a
   * bit-vector of the signed word width, applying the bit-vector operation,
   * and converting back. */
  template <typename Fn>
  smt_astt int_bitwise_unary(smt_astt a, Fn &&op)
  {
    const unsigned width = signed_size_type2()->get_width();
    auto a_bv = solver->mkInt2BV(width, expr(a));
    return wrap(solver->mkBV2Int(op(a_bv), true), mk_int_sort());
  }

  template <typename Fn>
  smt_astt int_bitwise_binary(smt_astt a, smt_astt b, Fn &&op)
  {
    const unsigned width = signed_size_type2()->get_width();
    auto a_bv = solver->mkInt2BV(width, expr(a));
    auto b_bv = solver->mkInt2BV(width, expr(b));
    return wrap(solver->mkBV2Int(op(a_bv, b_bv), true), mk_int_sort());
  }

  static smt_sortt from_camada_sort(const camada::SMTSortRef &sort)
  {
    using namespace camada;
    switch (sort->getSortKind())
    {
    case SMTSortKind::Bool:
      return new camada_sort(SMT_SORT_BOOL, sort, 1);
    case SMTSortKind::Int:
      return new camada_sort(SMT_SORT_INT, sort);
    case SMTSortKind::Real:
      return new camada_sort(SMT_SORT_REAL, sort);
    case SMTSortKind::BV:
      return new camada_sort(SMT_SORT_BV, sort, sort->getWidth());
    case SMTSortKind::FP:
      return new camada_sort(
        SMT_SORT_FPBV,
        sort,
        sort->getWidth(),
        sort->getFPSignificandWidth() + 1);
    case SMTSortKind::RM:
      return new camada_sort(SMT_SORT_FPBV_RM, sort, sort->getWidth());
    case SMTSortKind::BVFP:
      return new camada_sort(
        SMT_SORT_BVFP,
        sort,
        sort->getWidth(),
        sort->getFPSignificandWidth() + 1);
    case SMTSortKind::BVRM:
      return new camada_sort(SMT_SORT_BVFP_RM, sort, sort->getWidth());
    case SMTSortKind::Array:
    {
      unsigned index_width = 0;
      auto index_sort = sort->getIndexSort();
      if (index_sort->isBVSort())
        index_width = index_sort->getWidth();

      return new camada_sort(
        SMT_SORT_ARRAY,
        sort,
        index_width,
        from_camada_sort(sort->getElementSort()));
    }
    case SMTSortKind::Tuple:
      unsupported("tuple sort conversion without ESBMC type");
    case SMTSortKind::Function:
      unsupported("function sorts");
    case SMTSortKind::FXP:
      // ESBMC lowers fixedbv to a plain BV (mk_fbv_sort), so camada never
      // hands one back here.
      unsupported("fixed-point sorts");
    }

    unsupported("unknown Camada sort kind");
  }

  smt_astt wrap(const camada::SMTExprRef &value, smt_sortt sort)
  {
    if (sort->id == SMT_SORT_STRUCT)
      return new camada_tuple_ast(this, value, sort);
    return new camada_expr(this, value, sort);
  }

  template <typename Fn>
  smt_astt
  wrap_binary(smt_astt a, smt_astt b, Fn &&fn, smt_sortt sort = nullptr)
  {
    if (sort == nullptr)
      sort = a->sort;
    return wrap(fn(expr(a), expr(b)), sort);
  }

  camada::SMTExprRef
  make_index_expr(const camada::SMTSortRef &sort, uint64_t index)
  {
    if (sort->isBVSort())
      return solver->mkBVFromDec(static_cast<int64_t>(index), sort);
    if (sort->isIntSort())
      return solver->mkInt(static_cast<int64_t>(index));
    if (sort->isRealSort())
      return solver->mkReal(static_cast<int64_t>(index));
    unsupported("array index sort");
  }

  smt_astt fp_sign_test(smt_astt op, bool negative)
  {
    auto as_bv = solver->mkIEEEFPToBV(expr(op));
    auto sign = solver->mkBVExtract(
      op->sort->get_data_width() - 1, op->sort->get_data_width() - 1, as_bv);
    auto expected = solver->mkBVFromDec(negative ? 1 : 0, 1);
    return wrap(solver->mkEqual(sign, expected), boolean_sort);
  }
};

smt_astt camada_tuple_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  if (sort->id == SMT_SORT_ARRAY)
    return smt_ast::update(ctx, value, idx, idx_expr);

  assert(sort->id == SMT_SORT_STRUCT);
  assert(is_nil_expr(idx_expr));

  auto *cam_ctx = static_cast<camada_convt *>(ctx);
  const std::vector<type2tc> &members =
    struct_union_members(sort->get_tuple_type());
  std::vector<camada::SMTExprRef> fields;
  fields.reserve(members.size());
  for (std::size_t i = 0; i < members.size(); ++i)
  {
    if (i == idx)
      fields.push_back(to_solver_smt_ast<camada_expr>(value)->a);
    else
      fields.push_back(cam_ctx->solver->mkTupleSelect(a, i));
  }

  return cam_ctx->wrap(cam_ctx->solver->mkTuple(fields), sort);
}

smt_astt camada_tuple_ast::project(smt_solver_baset *ctx, unsigned int elem) const
{
  auto *cam_ctx = static_cast<camada_convt *>(ctx);
  const std::vector<type2tc> &members =
    struct_union_members(sort->get_tuple_type());
  assert(elem < members.size());
  const smt_sort *idx_sort = ctx->convert_sort(members[elem]);
  return cam_ctx->wrap(cam_ctx->solver->mkTupleSelect(a, elem), idx_sort);
}

smt_solver_baset *create_camada_solver(
  camada_backendt backend,
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api,
  bool oneshot = false)
{
  auto *solver = new camada_convt(ns, options, backend, oneshot);
  /* A one-shot script is flattened; only the interactive SMT-LIB backend
   * gets camada's native datatype tuples. */
  *tuple_api = (backend_supports_tuples(backend) && !oneshot)
                 ? static_cast<tuple_iface *>(solver)
                 : nullptr;

  /* A caller-named logic is a promise about what the external solver parses.
   * When it carries neither arrays nor floating-point (QF_BV, say), leave the
   * array and fp interfaces unset so create_solver() installs the flatteners
   * that lower both to bit-vectors before anything reaches the serializer. */
  const std::string forced_logic = options.get_option("smtlib-logic");
  const bool bv_only = !forced_logic.empty() &&
                       forced_logic.find("A") == std::string::npos &&
                       forced_logic.find("FP") == std::string::npos;
  *array_api = bv_only ? nullptr : solver;
  *fp_api = bv_only ? nullptr : solver;
  return solver;
}

} // namespace

smt_solver_baset *create_new_z3_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  return create_camada_solver(
    camada_backendt::z3, options, ns, tuple_api, array_api, fp_api);
}

smt_solver_baset *create_new_cvc5_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  return create_camada_solver(
    camada_backendt::cvc5, options, ns, tuple_api, array_api, fp_api);
}

smt_solver_baset *create_new_mathsat_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  return create_camada_solver(
    camada_backendt::mathsat, options, ns, tuple_api, array_api, fp_api);
}

smt_solver_baset *create_new_yices_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  return create_camada_solver(
    camada_backendt::yices, options, ns, tuple_api, array_api, fp_api);
}

smt_solver_baset *create_new_bitwuzla_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  return create_camada_solver(
    camada_backendt::bitwuzla, options, ns, tuple_api, array_api, fp_api);
}

/* The one-shot command processes a single task and exits; strategies that
 * solve repeatedly or incrementally cannot be served by it. */
void reject_incremental_strategies(const optionst &options)
{
  static const char *incompatible[] = {
    "incremental-bmc",
    "falsification",
    "k-induction",
    "k-induction-parallel",
    "termination",
    "smt-during-symex",
    "multi-property",
    "parallel-solving"};

  for (const char *opt : incompatible)
    if (options.get_bool_option(opt))
    {
      log_error(
        "--smtlib-oneshot-prog runs the solver once and cannot serve --{}; "
        "use a linked solver (e.g. --bitwuzla) for incremental strategies",
        opt);
      abort();
    }
}

smt_solver_baset *create_new_smtlib_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  /* --smtlib-oneshot-prog selects the write-a-file-and-run-a-program shape;
   * without it the script goes to an interactive solver or to --output. */
  const bool oneshot = !options.get_option("smtlib-oneshot-prog").empty();
  if (oneshot)
    reject_incremental_strategies(options);
  else if (!options.get_bool_option("smt-formula-only"))
    log_warning(
      "[smtlib] the smtlib interface solving is unstable. Please, "
      "use it with --smt-formula-only for production");

  return create_camada_solver(
    camada_backendt::smtlib,
    options,
    ns,
    tuple_api,
    array_api,
    fp_api,
    oneshot);
}
