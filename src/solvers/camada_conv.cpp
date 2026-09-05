#include <solvers/solve.h>
#include <solvers/smt_solver.h>
#include <solvers/oneshot_options.h>
#include <solvers/external_process_died.h>
#include <util/base/filesystem.h>

#include <util/arith/ieee_float.h>
#include <util/arith/mp_arith.h>

#include <camada/camada.h>
#include <camada/camadafeatures.h>
/* Camada's SMT-LIB backend drives the external solver with POSIX
 * fork/exec/setrlimit/select, so camada builds it -- and installs its header
 * -- only where that exists. */
#if CAMADA_HAVE_SMTLIB
#  include <camada/solvers/smtlibsolver.h>
#endif
#if CAMADA_HAVE_BITWUZLA
#  include <camada/solvers/bitwuzlasolver.h>
#endif
#if CAMADA_HAVE_MATHSAT
#  include <camada/solvers/mathsatsolver.h>
#endif
#if CAMADA_HAVE_YICES
#  include <camada/solvers/yicessolver.h>
#endif
#if CAMADA_HAVE_Z3
#  include <camada/solvers/z3solver.h>
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

#if CAMADA_HAVE_Z3
static void z3_error_handler(Z3_context c, Z3_error_code e)
{
  log_error("Z3 error {} encountered", Z3_get_error_msg(c, e));
  abort();
}

class esbmc_z3_solver : public camada::Z3Solver
{
public:
  /* camada v0.15 pins Z3 back to 4.13.3, whose z3::context has no move
   * constructor, so Z3Solver takes a z3::config (or nothing) instead of a
   * context. Mirror both shapes. */
  explicit esbmc_z3_solver(const camada::SolverConfig &config)
    : camada::Z3Solver(config)
  {
    setSolver(make_solver(camada::Z3Solver::context()));
  }

  explicit esbmc_z3_solver(z3::config &cfg, const camada::SolverConfig &config)
    : camada::Z3Solver(cfg, config)
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
  /* Camada builds the context from SolverConfig::Logic already; the
   * subclass exists only to rebuild it in push-pop mode, which
   * --smt-during-symex needs and SolverConfig has no field for. */
  explicit esbmc_yices_solver(
    bool enable_push_pop,
    const camada::SolverConfig &config)
    : camada::YicesSolver(config)
  {
    if (enable_push_pop)
      recreateContextWithConfig(config.Logic.c_str(), configure_push_pop);
  }

private:
  static void configure_push_pop(ctx_config_t *config)
  {
    yices_set_config(config, "mode", "push-pop");
  }
};
#endif

#if CAMADA_HAVE_MATHSAT
class esbmc_mathsat_solver : public camada::MathSATSolver
{
public:
  explicit esbmc_mathsat_solver(
    const msat_config &msat_cfg,
    const camada::SolverConfig &config)
    : camada::MathSATSolver(msat_cfg, config)
  {
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

/* Arrays are camada's to encode: `Native` there does not mean "require the
 * theory" -- camada uses the backend's own where it has one and lowers to
 * Ackermann congruence axioms otherwise, so it already picks per backend.
 *
 * Tuples on the SMT-LIB wire are different, and `Native` is the wrong default
 * for them. It emits `(declare-datatypes ...)`, which only z3 and cvc5 accept
 * (camada.h:124-131) -- and it emits them under the logic pick_logic() names,
 * QF_AUFBV, which admits no datatypes at all. A conforming solver rejects the
 * declaration:
 *
 *   (error "logic does not support algebraic datatypes")
 *
 * That reply is not `success`, so camada's ack reader drops the child before
 * `(check-sat)` is ever sent -- silently, since a protocol error is
 * indistinguishable there from an ack timeout. The visible symptom was an
 * auxiliary model solver reported as "unavailable" (regression/bitwuzllob/
 * mono-diverging-model), which in turn made the diverging-model check below
 * unreachable: with no verdict from the model solver there is nothing to
 * compare against.
 *
 * `Camada` lowers tuples to per-field BV/Bool symbols before anything reaches
 * the wire, so the script stays inside the declared logic and parses in any
 * standard SMT-LIB v2 solver. It is also what the non-native backends already
 * use, so this is not a new code path. */
/* camada v0.18 gathers what used to be separate constructor arguments --
 * array encoding, tuple encoding, logic, unsat-assumption mode -- into one
 * SolverConfig. Build it in one place so every backend gets the same choices.
 *
 * Logic is left empty here: only the backends that name one (MathSAT, Yices,
 * SMT-LIB) set it, and they do so from pick_logic() at their own call site.
 *
 * UseUnsatAssumptions stays false (the v0.18 default): ESBMC never calls
 * getUnsatAssumptions(), and producing cores costs solve time on every check
 * in the backends that must track assumption participation. */
camada::SolverConfig pick_solver_config(const optionst &)
{
  camada::SolverConfig config;
  config.Arrays = camada::ArrayEncoding::Native;
  config.Tuples = camada::TupleEncoding::Camada;
  return config;
}

/* Same, plus the concrete logic the SMT-LIB backends announce. */
[[maybe_unused]] camada::SolverConfig
smtlib_config(const optionst &options, const std::string &logic)
{
  camada::SolverConfig config = pick_solver_config(options);
  config.Logic = logic;
  return config;
}

/* maybe_unused: the backends that name a logic (MathSAT, Yices, SMT-LIB) are
 * each optional, so a build with none of them never calls this. */
[[maybe_unused]] std::string pick_logic(const optionst &options, bool native_fp)
{
  const bool has_quantifiers = options.get_bool_option("has-quantifiers");

  if (options.get_bool_option("int-encoding"))
    return has_quantifiers ? "AUFLIRA" : "QF_AUFLIRA";

  if (!native_fp)
    return has_quantifiers ? "AUFBV" : "QF_AUFBV";

  return has_quantifiers ? "AUFBVFP" : "QF_AUFBVFP";
}

#if CAMADA_HAVE_SMTLIB
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
      out.empty() ? "-" : out, smtlib_config(options, logic));

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
      camada::SMTLIBProcessTag{}, argv, smtlib_config(options, logic));

  return std::make_unique<camada::SMTLIBSolver>(
    camada::SMTLIBProcessTag{}, argv, out, smtlib_config(options, logic));
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
      formula_path, smtlib_config(options, logic));

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
    smtlib_config(options, logic));
}
#endif /* CAMADA_HAVE_SMTLIB */

} // namespace

/* Declared in solve.h: the selection table in solve.cpp points at these, so
 * they need external linkage. Everything above stays file-local. */

camada::SMTSolverRef create_esbmc_cvc5_solver(const optionst &options)
{
#if CAMADA_HAVE_CVC5
  return camada::createCVC5Solver(pick_solver_config(options));
#else
  (void)options;
  unsupported("CVC5 support in Camada");
#endif
}

camada::SMTSolverRef create_esbmc_bitwuzla_solver(const optionst &options)
{
#if CAMADA_HAVE_BITWUZLA
  return camada::createBitwuzlaSolver(pick_solver_config(options));
#else
  (void)options;
  unsupported("Bitwuzla support in Camada");
#endif
}

camada::SMTSolverRef create_esbmc_z3_solver(const optionst &options)
{
#if CAMADA_HAVE_Z3
  std::string z3_file = options.get_option("z3-debug-dump-file");
  const bool z3_debug = options.get_bool_option("z3-debug");
  const bool smtlib2_compliant = options.get_bool_option("smt-formula-only") ||
                                 options.get_bool_option("smt-formula-too");

  /* The config branch only fires for the z3-debug / smtlib-compliant
   * modes; the plain path uses the default context. */
  std::unique_ptr<esbmc_z3_solver> solver;
  if (!z3_debug && !smtlib2_compliant)
    solver = std::make_unique<esbmc_z3_solver>(pick_solver_config(options));
  else
  {
    z3::config cfg;
    if (z3_debug)
    {
      Z3_open_log(z3_file.empty() ? "z3.log" : z3_file.c_str());
      cfg.set("stats", "true");
      cfg.set("type_check", "true");
      cfg.set("well_sorted_check", "true");
    }

    cfg.set("smtlib2_compliant", "true");

    solver =
      std::make_unique<esbmc_z3_solver>(cfg, pick_solver_config(options));
  }
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

  auto solver = std::make_unique<esbmc_mathsat_solver>(
    config, smtlib_config(options, logic));
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
    options.get_bool_option("smt-during-symex"), smtlib_config(options, logic));
#else
  (void)options;
  unsupported("Yices support in Camada");
#endif
}

namespace
{
/* Sort widths, read straight from camada rather than from a copy kept in the
 * camada. Only a native FP sort carries FP structure there; mk_bvfp_sort
 * builds a plain bit-vector, whose significand and exponent are never read
 * back. */
static unsigned sort_fp_ew(smt_sortt s)
{
  return s->getFPExponentWidth();
}

static unsigned sort_fp_sw(smt_sortt s)
{
  return s->getFPSignificandWidth();
}

} // namespace

smt_resultt smt_solver_baset::dec_solve()
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
smt_resultt smt_solver_baset::oneshot_dec_solve()
{
#if !CAMADA_HAVE_SMTLIB
  /* `oneshot` is only ever set by the SMT-LIB factory, which does not exist
   * here, so nothing can reach this. */
  unsupported("the SMT-LIB backend on this platform");
#else
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
#endif
}

void smt_solver_baset::assert_ast(smt_astt a)
{
  solver->addConstraint(a);
}

/* A write-only SMT-LIB solver emits the script and never reads anything back,
 * so get()/l_get() have nothing to answer with; in one-shot mode the model
 * comes from the auxiliary interactive solver, which is only there while it
 * is live. Every linked backend always has a model. Without this, --result-only
 * runs still reach the readout paths and die on "write-only mode does not
 * support get*". */
bool smt_solver_baset::has_model() const
{
#if CAMADA_HAVE_SMTLIB
  if (oneshot)
  {
    const auto *smtlib =
      dynamic_cast<const camada::SMTLIBSolver *>(solver.get());
    return smtlib && smtlib->oneShotModelSolverLive();
  }
#endif
  return !streams_script;
}

/* Non-null only when the model comes from an external one-shot model
 * solver, whose disappearance must be reported rather than defaulted. */
const char *smt_solver_baset::oneshot_label() const
{
  return oneshot ? "smtlib" : nullptr;
}

tvt smt_solver_baset::get_bool(smt_astt a)
{
  const auto value = a;

  /* Quantified terms are handed to the solver like any other: camada evaluates
   * them, and getBool's SMTResult already models the case where it cannot --
   * unwrap_model_result below reports that and yields TV_UNKNOWN.
   *
   * An earlier version bailed out here whenever the term was a quantifier or
   * its dump contained "(forall "/"(exists ", returning TV_UNKNOWN without
   * asking. That produced the same value the failure path would have, so it
   * protected nothing, and it discarded values camada could supply: with it in
   * place, counterexample attribution had no concrete booleans to classify and
   * regression/z3/github_6191_attribution lost its "genuine" line. */
  auto result = unwrap_model_result(
    solver->getBool(value), "boolean model value", oneshot_label());
  if (!result)
    return tvt(tvt::TV_UNKNOWN);

  return tvt(*result);
}

BigInt smt_solver_baset::get_bv(smt_astt a, bool is_signed)
{
  const auto exp = a;
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

BigInt smt_solver_baset::get_fxp(smt_astt a)
{
  auto result = unwrap_model_result(
    solver->getFXP(a), "fixed-point model value", oneshot_label());
  return result ? binary2integer(result->RawBits, result->IsSigned) : BigInt(0);
}

ieee_floatt smt_solver_baset::get_fpbv(smt_astt a)
{
  auto model_result = unwrap_model_result(
    solver->getFPInBin(a), "floating-point model value", oneshot_label());
  if (!model_result)
    return ieee_floatt(
      ieee_float_spect(sort_fp_sw(a->Sort), sort_fp_ew(a->Sort)));

  std::string bits = *model_result;
  const auto ew = sort_fp_ew(a->Sort);
  const auto sw = sort_fp_sw(a->Sort);
  ieee_floatt result(ieee_float_spect(sw, ew));
  result.unpack(binary2integer(bits, false));
  return result;
}

bool smt_solver_baset::get_rational(
  smt_astt a,
  BigInt &numerator,
  BigInt &denominator)
{
  auto result = unwrap_model_result(
    solver->getRational(a), "rational model value", oneshot_label());
  if (!result)
    return false;

  numerator = BigInt(result->first.c_str(), 10);
  denominator = BigInt(result->second.c_str(), 10);
  return true;
}

expr2tc smt_solver_baset::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  auto idx = make_index_expr(array->Sort->getIndexSort(), index);
  auto elem = solver->getArrayElement(array, idx);
  return get_by_ast(subtype, elem);
}

smt_astt smt_solver_baset::mk_add(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithAdd(a, b) : solver->mkBVAdd(a, b);
}

smt_astt smt_solver_baset::mk_sub(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithSub(a, b) : solver->mkBVSub(a, b);
}
smt_astt smt_solver_baset::mk_mul(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithMul(a, b) : solver->mkBVMul(a, b);
}
smt_astt smt_solver_baset::mk_mod(smt_astt a, smt_astt b)
{
  return solver->mkArithMod(a, b);
}
smt_astt smt_solver_baset::mk_bvumod(smt_astt a, smt_astt b)
{
  return solver->mkBVURem(a, b);
}
smt_astt smt_solver_baset::mk_div(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithDiv(a, b) : solver->mkBVUDiv(a, b);
}
smt_astt smt_solver_baset::mk_bvsdiv(smt_astt a, smt_astt b)
{
  return solver->mkBVSDiv(a, b);
}
smt_astt smt_solver_baset::mk_shl(smt_astt a, smt_astt b)
{
  return solver->mkArithShl(a, b);
}
smt_astt smt_solver_baset::mk_bvashr(smt_astt a, smt_astt b)
{
  return solver->mkBVAshr(a, b);
}
smt_astt smt_solver_baset::mk_neg(smt_astt a)
{
  auto ea = a;
  return ea->isArithSort() ? solver->mkArithNeg(ea) : solver->mkBVNeg(ea);
}
smt_astt smt_solver_baset::mk_bvnot(smt_astt a)
{
  if (int_encoding)
  {
    const unsigned width = signed_size_type2()->get_width();
    return solver->mkBV2Int(solver->mkBVNot(solver->mkInt2BV(width, a)), true);
  }
  return solver->mkBVNot(a);
}
smt_astt smt_solver_baset::mk_bvxor(smt_astt a, smt_astt b)
{
  if (int_encoding)
    return int_bitwise_binary(
      a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
        return solver->mkBVXor(l, r);
      });
  return solver->mkBVXor(a, b);
}
smt_astt smt_solver_baset::mk_bvor(smt_astt a, smt_astt b)
{
  if (int_encoding)
    return int_bitwise_binary(
      a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
        return solver->mkBVOr(l, r);
      });
  return solver->mkBVOr(a, b);
}
smt_astt smt_solver_baset::mk_bvand(smt_astt a, smt_astt b)
{
  if (int_encoding)
    return int_bitwise_binary(
      a, b, [this](const camada::SMTExprRef &l, const camada::SMTExprRef &r) {
        return solver->mkBVAnd(l, r);
      });
  return solver->mkBVAnd(a, b);
}
smt_astt smt_solver_baset::mk_implies(smt_astt a, smt_astt b)
{
  return solver->mkImplies(a, b);
}
smt_astt smt_solver_baset::mk_or(smt_astt a, smt_astt b)
{
  return solver->mkOr(a, b);
}
smt_astt smt_solver_baset::mk_not(smt_astt a)
{
  return solver->mkNot(a);
}
smt_astt smt_solver_baset::mk_lt(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithLt(a, b) : solver->mkBVUlt(a, b);
}
smt_astt smt_solver_baset::mk_bvult(smt_astt a, smt_astt b)
{
  return solver->mkBVUlt(a, b);
}
smt_astt smt_solver_baset::mk_gt(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithGt(a, b) : solver->mkBVUgt(a, b);
}
smt_astt smt_solver_baset::mk_bvsgt(smt_astt a, smt_astt b)
{
  return solver->mkBVSgt(a, b);
}
smt_astt smt_solver_baset::mk_le(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithLe(a, b) : solver->mkBVUle(a, b);
}
smt_astt smt_solver_baset::mk_bvule(smt_astt a, smt_astt b)
{
  return solver->mkBVUle(a, b);
}
smt_astt smt_solver_baset::mk_ge(smt_astt a, smt_astt b)
{
  return a->isArithSort() ? solver->mkArithGe(a, b) : solver->mkBVUge(a, b);
}
smt_astt smt_solver_baset::mk_bvsge(smt_astt a, smt_astt b)
{
  return solver->mkBVSge(a, b);
}
smt_astt smt_solver_baset::mk_neq(smt_astt a, smt_astt b)
{
  return solver->mkNot(solver->mkEqual(a, b));
}
smt_astt smt_solver_baset::mk_select(smt_astt a, smt_astt b)
{
  return solver->mkArraySelect(a, b);
}
smt_astt smt_solver_baset::mk_int2real(smt_astt a)
{
  return solver->mkInt2Real(a);
}

smt_sortt smt_solver_baset::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  auto sort = solver->mkBVSort(ew + sw + 1);
  return sort;
}

smt_sortt smt_solver_baset::mk_bvfp_rm_sort()
{
  auto sort = solver->mkBVSort(3);
  return sort;
}

smt_sortt smt_solver_baset::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  auto sort = solver->mkFPSort(ew, sw, fp_encoding());
  return sort;
}

smt_sortt smt_solver_baset::mk_fpbv_rm_sort()
{
  return solver->mkRMSort(fp_encoding());
}

smt_astt smt_solver_baset::mk_smt_int(const BigInt &theint)
{
  return solver->mkInt(integer2string(theint, 10));
}

smt_astt smt_solver_baset::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  return solver->mkBVFromBin(integer2binary(theint, s->getWidth()), s);
}

smt_astt smt_solver_baset::mk_smt_fpbv(const ieee_floatt &thereal)
{
  std::string bits = integer2binary(thereal.pack(), thereal.spec.width());
  return solver->mkFPFromBin(bits, thereal.spec.e, fp_encoding());
}

smt_astt smt_solver_baset::mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw)
{
  return solver->mkNaN(sgn, ew, sw, fp_encoding());
}

smt_astt smt_solver_baset::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  return solver->mkInf(sgn, ew, sw, fp_encoding());
}

smt_astt smt_solver_baset::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  return solver->mkRM(to_camada_rm(rm), fp_encoding());
}

smt_astt
smt_solver_baset::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return solver->mkFPSub(lhs, rhs, rm);
}
smt_astt
smt_solver_baset::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return solver->mkFPDiv(lhs, rhs, rm);
}
smt_astt smt_solver_baset::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return solver->mkFPSqrt(rd, rm);
}

smt_astt smt_solver_baset::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return solver->mkFPGt(lhs, rhs);
}
smt_astt smt_solver_baset::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return solver->mkFPGe(lhs, rhs);
}
smt_astt smt_solver_baset::mk_smt_fpbv_is_nan(smt_astt op)
{
  return solver->mkFPIsNaN(op);
}
smt_astt smt_solver_baset::mk_smt_fpbv_is_normal(smt_astt op)
{
  return solver->mkFPIsNormal(op);
}
smt_astt smt_solver_baset::mk_smt_fpbv_is_negative(smt_astt op)
{
  return fp_sign_test(op, true);
}
smt_astt smt_solver_baset::mk_smt_fpbv_is_positive(smt_astt op)
{
  return fp_sign_test(op, false);
}
smt_astt smt_solver_baset::mk_smt_fpbv_abs(smt_astt op)
{
  return solver->mkFPAbs(op);
}

smt_astt smt_solver_baset::mk_from_fp_to_bv(smt_astt op)
{
  return solver->mkIEEEFPToBV(op);
}

smt_astt smt_solver_baset::mk_array_symbol(
  const std::string &name,
  smt_sortt sort,
  smt_sortt)
{
  return solver->mkSymbol(name, sort);
}

smt_sortt smt_solver_baset::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    smt_sortt subtypesort = convert_sort(arrtype.subtype);
    smt_sortt d = mk_int_bv_sort(make_array_domain_type(arrtype)->get_width());
    return solver->mkArraySort(d, subtypesort);
  }

  const std::vector<type2tc> &members = struct_union_members(type);
  std::vector<camada::SMTSortRef> field_sorts;
  field_sorts.reserve(members.size());
  for (const auto &member : members)
    field_sorts.push_back(convert_sort(member));

  return solver->mkTupleSort(field_sorts);
}

smt_astt
smt_solver_baset::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  // If it's a floatbv, convert it to bv first so callers extracting bytes
  // out of structs/unions containing floats encode against the IEEE bit
  // pattern instead of triggering a sort mismatch.
  if (a->Sort->isFPSort())
    a = mk_from_fp_to_bv(a);
  return solver->mkBVExtract(high, low, a);
}

smt_astt smt_solver_baset::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  return solver->mkBVSignExt(topwidth, a);
}

smt_astt smt_solver_baset::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  return solver->mkBVZeroExt(topwidth, a);
}

smt_astt smt_solver_baset::tuple_create(const expr2tc &structdef)
{
  const constant_struct2t &strct = to_constant_struct2t(structdef);

  /* Build against the sort the *type* dictates, not the one the members happen
   * to convert to. A constant_struct can carry a member already lowered to an
   * integer where the struct type declares a pointer -- std::type_info's
   * __impl field does exactly that. Synthesising the tuple sort from the
   * converted members then yields <tuple, BV> where a symbol of the same type
   * gets <tuple, tuple> from convert_sort(), and the two are not comparable:
   * mkEqual rejects the assignment (esbmc/esbmc#6310). The declared type is the
   * single source of truth, so a member whose sort disagrees is re-converted at
   * the field's type. */
  const std::vector<type2tc> &member_types = struct_union_members(
    is_pointer_type(structdef->type) ? pointer_struct : structdef->type);
  const auto &want = convert_sort(structdef->type)->getTupleElementSorts();

  std::vector<camada::SMTExprRef> fields;
  fields.reserve(strct.datatype_members.size());
  for (std::size_t i = 0; i < strct.datatype_members.size(); ++i)
  {
    const expr2tc &member = strct.datatype_members[i];
    smt_astt m = convert_ast(member);
    if (i < want.size() && m->Sort != want[i] && i < member_types.size())
      m = convert_ast(typecast2tc(member_types[i], member));
    fields.push_back(m);
  }

  return solver->mkTuple(fields);
}

smt_astt smt_solver_baset::tuple_fresh(smt_sortt s, std::string name)
{
  if (name.empty())
    name = mk_fresh_name("smt_solver_baset::tuple_fresh");
  return solver->mkSymbol(name, s);
}

smt_astt smt_solver_baset::tuple_array_create(
  const type2tc &array_type,
  smt_astt *input_args,
  bool const_array,
  smt_sortt domain)
{
  const array_type2t &arrtype = to_array_type(array_type);
  smt_sortt elem_sort = mk_struct_sort(arrtype.subtype);
  smt_sortt array_sort = solver->mkArraySort(domain, elem_sort);

  if (const_array)
  {
    return solver->mkArrayConst(domain, *input_args);
  }

  assert(!is_nil_expr(arrtype.array_size));
  assert(is_constant_int2t(arrtype.array_size));

  auto result = solver->mkSymbol(
    mk_fresh_name("smt_solver_baset::tuple_array_create"), array_sort);
  auto domain_sort = domain;

  for (std::size_t i = 0; i < to_constant_int2t(arrtype.array_size).as_ulong();
       ++i)
  {
    result = solver->mkArrayStore(
      result, make_index_expr(domain_sort, i), input_args[i]);
  }

  return result;
}

smt_astt smt_solver_baset::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  if (name == "NULL")
    return null_ptr_ast;

  if (name == "INVALID")
    return invalid_ptr_ast;

  return solver->mkSymbol(name, s);
}

smt_astt smt_solver_baset::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return solver->mkSymbol(sym.get_symbol_name(), convert_sort(sym.type));
}

smt_astt smt_solver_baset::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  return convert_array_of(convert_ast(init_value), domain_width);
}

expr2tc smt_solver_baset::tuple_get(const expr2tc &expr)
{
  return tuple_get(expr->type, convert_ast(expr));
}

expr2tc smt_solver_baset::tuple_get(const type2tc &type, smt_astt sym)
{
  // Pointer types lower to pointer_struct in SMT; struct_union_members(type)
  // would throw on the raw pointer type, so dispatch the pointer case first.
  const std::vector<type2tc> &members =
    struct_union_members(is_pointer_type(type) ? pointer_struct : type);

  if (is_pointer_type(type))
  {
    smt_astt object = solver->mkTupleSelect(sym, 0);
    smt_astt offset = solver->mkTupleSelect(sym, 1);

    unsigned int num = get_bv(object, is_signedbv_type(members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, type);
  }

  std::vector<expr2tc> outmem;
  outmem.reserve(members.size());
  for (std::size_t i = 0; i < members.size(); ++i)
  {
    outmem.push_back(get_by_ast(members[i], solver->mkTupleSelect(sym, i)));
  }

  return constant_struct2tc(type, std::move(outmem));
}

expr2tc smt_solver_baset::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  return get_array_elem(array, index, get_flattened_array_subtype(subtype));
}

smt_astt smt_solver_baset::convert_array_of(
  smt_astt init_val,
  unsigned long domain_width)
{
  auto idx_sort = int_encoding
                    ? solver->mkIntSort()
                    : solver->mkBVSort(domain_width == 0 ? 1 : domain_width);
  auto value = solver->mkArrayConst(idx_sort, init_val);
  return value;
}

const std::string smt_solver_baset::solver_text()
{
  if (oneshot)
    return "one-shot '" + oneshot_prog + "'";
  return solver->getSolverNameAndVersion();
}

std::string smt_solver_baset::dump_smt()
{
  /* Camada's SMT-LIB backend streams the script to its sink as it is built
   * rather than buffering it, so there is nothing to hand back. Complete the
   * script with the (check-sat) that --smt-formula-only never reaches via
   * dec_solve() -- in write-only mode check() just emits it and answers
   * UNKNOWN -- then return empty so bmc.cpp does not reopen the same path
   * and overwrite what camada wrote (issue #6059). */
  if (streams_script)
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

void smt_solver_baset::print_model()
{
  solver->dumpModel();
}

smt_astt smt_solver_baset::mk_quantifier(
  bool is_forall,
  std::vector<smt_astt> lhs,
  smt_astt rhs)
{
  std::vector<camada::SMTExprRef> vars;
  vars.reserve(lhs.size());
  for (const auto &var : lhs)
    vars.push_back(var);

  auto q =
    is_forall ? solver->mkForall(vars, rhs) : solver->mkExists(vars, rhs);
  return q;
}

/* --fp2bv asks for floating-point encoded as bit-vectors, which is exactly
 * camada's FPEncoding::BV; let it bit-blast rather than swapping in ESBMC's
 * own software lowering. */
camada::FPEncoding smt_solver_baset::fp_encoding() const
{
  return options.get_bool_option("fp2bv") ? camada::FPEncoding::BV
                                          : camada::FPEncoding::Native;
}

camada::SMTExprRef smt_solver_baset::make_index_expr(
  const camada::SMTSortRef &sort,
  uint64_t index)
{
  if (sort->isBVSort())
    return solver->mkBVFromDec(static_cast<int64_t>(index), sort);
  if (sort->isIntSort())
    return solver->mkInt(static_cast<int64_t>(index));
  if (sort->isRealSort())
    return solver->mkReal(static_cast<int64_t>(index));
  unsupported("array index sort");
}

smt_astt smt_solver_baset::ast_update(
  smt_astt a,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr)
{
  if (a->Sort->isTupleSort())
  {
    assert(is_nil_expr(idx_expr));
    return solver->mkTupleUpdate(a, idx, value);
  }

  assert(a->Sort->isArraySort());

  expr2tc index;
  if (is_nil_expr(idx_expr))
  {
    size_t dom_width = int_encoding ? config.ansi_c.int_width
                                    : a->Sort->getIndexSort()->getWidth();
    index = constant_int2tc(unsignedbv_type2tc(dom_width), BigInt(idx));
  }
  else
  {
    index = idx_expr;
  }

  return solver->mkArrayStore(a, convert_ast(index), value);
}

smt_astt smt_solver_baset::ast_project(smt_astt a, unsigned int elem)
{
  assert(elem < a->Sort->getTupleElementSorts().size());
  return solver->mkTupleSelect(a, elem);
}

smt_astt smt_solver_baset::fp_sign_test(smt_astt op, bool negative)
{
  auto as_bv = solver->mkIEEEFPToBV(op);
  auto sign = solver->mkBVExtract(
    op->Sort->getWidth() - 1, op->Sort->getWidth() - 1, as_bv);
  auto expected = solver->mkBVFromDec(negative ? 1 : 0, 1);
  return solver->mkEqual(sign, expected);
}

/* Every backend below knows which camada solver it wants, so it builds it and
 * hands it over; there is no backend tag to switch on. Everything else the
 * solver needs it reads from `options` itself. */
std::unique_ptr<smt_solver_baset> make_solver(
  const optionst &options,
  const namespacet &ns,
  std::unique_ptr<camada::SMTSolver> solver,
  bool streams_script = false)
{
  return std::make_unique<smt_solver_baset>(
    ns, options, std::move(solver), streams_script);
}

std::unique_ptr<smt_solver_baset> create_linked_solver(
  const optionst &options,
  const namespacet &ns,
  camada_buildert build)
{
  return make_solver(options, ns, build(options));
}

#if CAMADA_HAVE_SMTLIB
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
#endif /* CAMADA_HAVE_SMTLIB */

std::unique_ptr<smt_solver_baset>
create_new_smtlib_solver(const optionst &options, const namespacet &ns)
{
#if !CAMADA_HAVE_SMTLIB
  /* ESBMC_ENABLE_smtlib is 0 here, so pick_solver rejects the backend before
   * anything can call this; the definition only has to link. */
  (void)options;
  (void)ns;
  unsupported("the SMT-LIB backend on this platform");
#else
  /* --smtlib-oneshot-prog selects the write-a-file-and-run-a-program shape;
   * without it the script goes to an interactive solver or to --output. */
  const bool oneshot = !options.get_option("smtlib-oneshot-prog").empty();
  if (!oneshot)
  {
    if (!options.get_bool_option("smt-formula-only"))
      log_warning(
        "[smtlib] the smtlib interface solving is unstable. Please, "
        "use it with --smt-formula-only for production");
    return make_solver(
      options,
      ns,
      create_esbmc_smtlib_solver(options),
      /*streams_script=*/true);
  }

  reject_incremental_strategies(options);

  const std::string oneshot_prog = options.get_option("smtlib-oneshot-prog");
  const std::string formula_path =
    oneshot_options::choose_formula_path(options, "smtlib");
  /* An explicit --smtlib-logic is the user telling us what the external
   * solver accepts; otherwise follow the encoding. */
  const std::string logic = options.get_option("smtlib-logic");
  return make_solver(
    options,
    ns,
    create_esbmc_oneshot_solver(
      options,
      "smtlib",
      formula_path,
      oneshot_prog,
      logic.empty() ? pick_logic(options, false) : logic));
#endif
}

smt_astt smt_solver_baset::mk_eq(smt_astt a, smt_astt b)
{
  return solver->mkEqual(a, b);
}

smt_sortt smt_solver_baset::mk_int_sort()
{
  return solver->mkIntSort();
}

smt_astt smt_solver_baset::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  return solver->mkSymbol(name, s);
}
