#include <csignal>
#include <memory>
#ifdef _WIN32
#  include <windows.h>
#endif
#include <sys/types.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <chrono>

#ifndef _WIN32
#  include <unistd.h>
#  include <sched.h>
#else
#  include <windows.h>
#  include <winbase.h>
#  undef ERROR
#  undef small
#endif

#include <filesystem>
#include <fmt/format.h>
#include <regex>
#include <ac_config.h>
#include <esbmc/bmc.h>
#include <esbmc/property_report.h>
#include <fstream>
#include <goto-programs/goto_loops.h>
#include <goto-symex/trace/build_goto_trace.h>
#include <goto-symex/symex_invariant.h>
#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/equation/features.h>
#include <goto-symex/trace/sarif.h>
#include <goto-symex/equation/symex_symmetry.h>
#include <goto-symex/trace/xml_goto_trace.h>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <solvers/smt/smt_conv.h>
#include <sstream>
#include <util/base/i2string.h>
#include <irep2/irep2.h>
#include <util/irep/location.h>

#include <util/irep/migrate.h>
#include <util/base/cwe_mapping.h>
#include <util/symtab/show_symbol_table.h>
#include <util/base/time_stopping.h>
#include <util/ssa/cache.h>
#include <util/ssa/fingerprint.h>
#include <util/ssa/proof_cache.h>
#include <esbmc/globals.h>
#include <atomic>
#include <vector>
#include <nlohmann/json.hpp>

static std::string ctest_output_dir(const optionst &options)
{
  std::string dir = options.get_option("ctest-output-dir");
  return dir.empty() ? ctest_generator::default_output_dir : dir;
}

static std::string pytest_output_dir(const optionst &options)
{
  std::string dir = options.get_option("pytest-output-dir");
  return dir.empty() ? pytest_generator::default_output_dir : dir;
}

// ---- DEFINED HERE, and deliberately ----
//
// These five are written and read ONLY by this file: the re-solve happens in
// multi_property_check's job loop and the numbers are printed by
// report_coverage, both below. goto_coverage.cpp neither sets nor reads them.
// Putting the definitions beside their only users follows the precedent
// immediately below (goto_functionst's four statics live here for the same
// reason) and means a reader who finds one finds all of them.
std::set<std::pair<std::string, std::string>>
  goto_coveraget::arith_revert_only_paths;
std::atomic<size_t> goto_coveraget::arith_resolve_queries{0};
std::atomic<size_t> goto_coveraget::arith_resolve_replaced{0};
std::atomic<size_t> goto_coveraget::arith_resolve_ms{0};
std::atomic<size_t> goto_coveraget::arith_conditions_seen{0};
std::atomic<size_t> goto_coveraget::arith_revert_only_suppressed{0};
std::atomic<size_t> goto_coveraget::verdicts_preserved{0};

std::unordered_set<std::string> goto_functionst::reached_claims;
std::unordered_multiset<std::string> goto_functionst::reached_mul_claims;
std::mutex goto_functionst::reached_claims_mutex;
std::mutex goto_functionst::reached_mul_claims_mutex;
std::mutex goto_functionst::clear_claims_mutex;
/* Coverage-completeness bookkeeping. File-scope rather than shared state on
 * goto_functionst: nothing outside this file produces or consumes it, and the
 * driver prints it once after the last [Coverage] block. Reset per
 * multi_property_check so a k-induction / incremental run reports the last
 * pass rather than the sum of every k step. */
static std::atomic<size_t> undecided_cov_goals{0};
static std::set<std::string> cov_incomplete_reasons;
// Claims a coverage run solved but does not report, because it reports no
// violations at all. Accumulated across passes: a violation found at one k
// step stays true.
static std::set<std::string> cov_suppressed_violations;
static std::mutex cov_report_mutex;
// Set once a [Coverage] block has actually been produced. Modes that never
// run BMC (--show-vcc, --program-only) would otherwise close with a
// completeness verdict over a measurement that never happened.
static std::atomic<bool> cov_block_reported{false};

// Record why the coverage measurement is not exhaustive, so the reported
// percentage can be qualified rather than silently understated.
void note_cov_incomplete(const std::string &reason)
{
  std::lock_guard lock(cov_report_mutex);
  cov_incomplete_reasons.insert(reason);
}

// Record a claim a coverage run proved violated but does not report.
static void note_cov_suppressed_violation(const std::string &claim)
{
  std::lock_guard lock(cov_report_mutex);
  cov_suppressed_violations.insert(claim);
}

// As above, for a specific goal whose reachability was never decided.
static void note_undecided_cov_goal(const std::string &reason)
{
  undecided_cov_goals++;
  note_cov_incomplete(reason);
}

/// Whether the k-step strategy driver owns the run's property table: it clears
/// the store before its first phase, every phase records into it, and it is
/// printed once where the run concludes (W3b of
/// docs/roadmap/multi-property-strategy-plan.md).
static bool strategy_owns_property_table(const optionst &options)
{
  return options.get_bool_option("k-step-property-table");
}

/// Whether this phase may report a --multi-property result of its own. Under a
/// k-step strategy it may not: its phases deepen the search rather than
/// conclude it, and the strategy reports for all of them.
static bool reports_multi_property_verdict(const optionst &options)
{
  return options.get_bool_option("multi-property") &&
         !strategy_owns_property_table(options);
}

/// Record what a solver's UNSAT earned \p property. Only a proof needs backing:
/// where withholds_proofs() says the phase cannot make one, the claim stays
/// undecided. A vacuous discharge is not a proof but a diagnosis of the path,
/// which a bounded round establishes as well as a conclusive one, so it is
/// recorded either way -- and it has to be, since a k-step run reports the
/// vacuity nowhere else.
static void record_discharge(
  bool withhold_proofs,
  const std::string &property,
  property_verdictt verdict,
  const property_locationt &loc,
  const std::string &note = "")
{
  if (withhold_proofs && verdict == property_verdictt::Passed)
    return;

  goto_functionst::property_verdicts.record(property, verdict, loc, note);
}

/// The closing line of a report that does not account for every property.
static void print_partial_report_note()
{
  log_result(
    "This report is partial: the run stopped before every property reached "
    "a verdict, so properties are missing above, and a passing verdict "
    "holds only for the thread interleavings explored. Raise "
    "--multi-property-interleavings, or drop --multi-fail-fast, to check "
    "further.");
}

/// Whether the table a k-step run accumulated says the program holds. The
/// run's verdict follows its table, so a row left undecided -- never settled
/// within max-k, or discharged only vacuously -- is not a proof, and the phase
/// that closes the search must not call it one. Vacuously true elsewhere: a
/// single run's phase speaks for itself.
static bool run_fully_proved(const optionst &options)
{
  return !strategy_owns_property_table(options) ||
         goto_functionst::property_verdicts.all_passed();
}

/// Whether a phase left properties undecided that a later one must not take
/// for proved: it skipped some claim, died before its per-claim loop, or only
/// emitted a formula.
static bool
phase_left_properties_undecided(bool report_incomplete, smt_resultt res)
{
  return report_incomplete || res == P_ERROR || res == P_SMTLIB;
}

/// Whether this phase is a k-step strategy's base case, which opens and
/// completes a round and discharges a claim only within the current k
/// (withholds_proofs()).
static bool is_bounded_round(const optionst &options)
{
  return options.get_bool_option("base-case") &&
         strategy_owns_property_table(options);
}

// ---- HOW OFTEN ONE CLAIM KEY WAS DECIDED, AND WHETHER IT CHANGED ITS MIND
// ----
//
// File-local for the same reason as the block above: the counting happens in
// multi_property_check's job loop and the printing in report_coverage, both in
// this file, and nothing else reads them.
//
// `Verdicts Preserved` already watches ONE direction -- a decision replaced by
// a non-decision. Its own header says a non-zero value "also means the same
// claim key was solved more than once". The converse is false, and that is the
// hole: MEASURED on notes/coverage/poc/P28_MapMin.sol, --solidity-max-tx 2,
// 4 instrumented paths and 8 VCCs, the key `take:path:15` was solved twice and
// got BOTH verdicts --
//
//     take:path:15   solved 2x   verdicts PASSED then FAILED
//     Verdicts Preserved: 0
//
// -- while the counter that is supposed to notice read zero, because the second
// solve DID return a verdict. A decision was contradicted and nothing recorded
// it.
//
// WHY THAT PARTICULAR CONTRADICTION IS NOT AN ERROR. The two solves are two
// INSTANCES of one path: the transaction body is emitted once per transaction,
// so the same assert instruction is reached once per transaction, and a path
// guarded by state that only an earlier transaction can establish holds in tx 1
// and is refuted in tx 2. The path's verdict is the DISJUNCTION over its
// instances, which is exactly what the existing rule computes (F is final, and
// P upgrades to F). Aborting on it -- the literal reading of "one key, two
// verdicts, therefore fatal" -- would abort every multi-transaction run for
// doing the one thing a multi-transaction harness exists to do.
//
// SO THE FATAL IS THE MULTIPLICITY, NOT THE DISAGREEMENT. One key may be
// decided at most once per transaction. More than that is not the disjunction,
// it is the duplicate instrumentation the write site's own comment calls "a
// SECOND defect ... not fixed here": the same claim emitted at more than one
// site, paying for solves that buy nothing and giving one path several chances
// to be witnessed. That is checkable against a number the run already knows.
namespace
{
std::mutex path_cov_solve_count_mutex;
std::map<std::string, size_t> path_cov_solve_count;
// A DECIDED verdict replaced by a DIFFERENT decided verdict (today only
// P -> F). Legitimate, and counted rather than silent: it is the event
// `Verdicts Preserved` cannot see, and a run in which it never happens must be
// distinguishable from one in which it happened forty times.
std::atomic<size_t> path_cov_verdict_upgrades{0};
// Solves of a key beyond its first. Bounded above by (transactions - 1) per
// key when the instrumentation is sound.
std::atomic<size_t> path_cov_extra_solves{0};
// The largest multiplicity seen, and the key that had it. Printed so the line
// is a measurement rather than a pass/fail.
size_t path_cov_max_solves = 0;
std::string path_cov_max_solves_key;
// How many times one key MAY be decided. Set once, from the transaction bound,
// before the job loop starts. 0 means the check is not armed (not a path
// coverage run).
size_t path_cov_allowed_solves = 0;
// Where the ceiling came from. The refusal message must not say "this run
// explores N transaction(s)" when N is an override -- that sentence would be
// FALSE on exactly the runs the override exists to produce, and a diagnostic
// that misstates the run is worse than none.
std::string path_cov_allowed_solves_origin;
} // namespace

bmct::bmct(
  goto_functionst &funcs,
  optionst &opts,
  contextt &_context,
  foundry_generator *ext_foundry_gen)
  : options(opts),
    context(_context),
    ns(context),
    foundry_gen_ext(ext_foundry_gen)
{
  interleaving_number = 0;
  interleaving_failed = 0;

  // The Python frontend hides functions imported from the user's own modules
  // too, so there a hidden body does not mean "our operational model" and the
  // report must not demote them (same caveat as remove_library_assertions).
  if (config.language.lid != language_idt::PYTHON)
    library_files = collect_library_assertion_files(funcs);

  ltl_results_seen[ltl_res_bad] = 0;
  ltl_results_seen[ltl_res_failing] = 0;
  ltl_results_seen[ltl_res_succeeding] = 0;
  ltl_results_seen[ltl_res_good] = 0;

  // The next block will initialize the algorithms used for the analysis.
  {
    // Run cache if user has specified the option
    if (
      !options.get_bool_option("no-cache-asserts") &&
      !options.get_bool_option("forward-condition") &&
      !options.get_bool_option("k-induction") &&
      !options.get_bool_option("ltl"))
      // Store the set between runs
      algorithms.emplace_back(
        std::make_unique<assertion_cache>(get_ssa_caching_db()));

    if (opts.get_bool_option("no-slice"))
      algorithms.emplace_back(std::make_unique<simple_slice>());
    else
      algorithms.emplace_back(std::make_unique<symex_slicet>(options));

    // Runs after slicing so it only decorates surviving max/min folds and
    // cannot resurrect assignments the slicer dropped.
    if (!opts.get_bool_option("no-symmetry-breaking"))
      algorithms.emplace_back(std::make_unique<symmetry_breakingt>());

    if (opts.get_bool_option("ssa-features-dump"))
      algorithms.emplace_back(std::make_unique<ssa_features>());
  }

  if (options.get_bool_option("smt-during-symex"))
  {
    runtime_solver = std::unique_ptr<smt_convt>(create_solver("", ns, options));

    symex = std::make_unique<reachability_treet>(
      funcs,
      ns,
      options,
      std::make_shared<runtime_encoded_equationt>(ns, *runtime_solver),
      _context);
  }
  else
  {
    symex = std::make_unique<reachability_treet>(
      funcs,
      ns,
      options,
      std::make_shared<symex_target_equationt>(ns),
      _context);
  }
}

bmct::~bmct() = default;

void bmct::successful_trace(const symex_target_equationt &eq [[maybe_unused]])
{
  if (options.get_bool_option("result-only"))
    return;

  std::string witness_graphml_output =
    options.get_option("witness-output-graphml");
  std::string witness_yaml_output = options.get_option("witness-output-yaml");

  goto_tracet goto_trace;
  if (witness_graphml_output != "")
    correctness_graphml_goto_trace(options, ns, goto_trace);

  if (witness_yaml_output != "")
    correctness_yaml_goto_trace(options, ns, goto_trace);

  // On a successful verification there is no error trace, but dead-store
  // advisories (CWE-563) are still valid and must reach SARIF. Emit an
  // advisory-only document; guarded so flag-off runs write nothing new.
  if (
    !dead_store_advisories.empty() &&
    !options.get_option("sarif-output").empty())
  {
    sarif_goto_trace(options, ns, goto_trace, dead_store_advisories);
    dead_store_sarif_written = true;
  }
}

// Obligations the schema discharges from the havoc'd state on purpose: does
// the body preserve the invariant, does it respect the assigns clause. Both
// are statements about the annotation rather than about a reachable state, so
// a model falsifying one is the finding the mode exists to report.
//
// The base case is not among them. It is checked from the state the loop is
// entered in, which for an outermost loop is concrete -- and the generic test
// already leaves it alone there, since no havoc precedes it. An inner loop's
// base case sits inside the outer body, downstream of the outer havoc, where
// the entry state is arbitrary: `s == i` holds at every real entry of the
// inner loop below and is still refuted there.
//
//   while (i < 3) { __ESBMC_loop_invariant(j >= 0 && s == i); ... s++; i++; }
//
// Exempting it by property name would pin that false alarm as a verdict, so it
// is left to the generic rule (issue #7480).
static bool is_loop_invariant_obligation(const locationt &location)
{
  const std::string property = location.property().as_string();
  return property == "invariant-inductive-step" ||
         property == "assigns compliance";
}

// Attached to a claim downgraded from Failed to Unknown because the only trace
// violating it runs through a --loop-invariant-check havoc (issue #7480).
static const char *const weak_invariant_note =
  "loop invariant too weak to prove this claim: the counterexample is against "
  "the havoc abstraction, not a reachable state of the program";

void bmct::record_satisfiable_claim(
  const claim_slicer &claim,
  const property_locationt &loc,
  bool inductive_step,
  symex_target_equationt &local_eq)
{
  // Neither answer refutes the program. An inductive-step run starts from an
  // arbitrary state, and a claim downstream of a loop-invariant havoc is
  // checked against the invariant's over-approximation, so a model witnesses
  // the annotation being too weak rather than a reachable state (#7480).
  if (inductive_step)
  {
    // Interim under a k-step strategy: the loop goes on to k+1, where the step
    // may well prove the claim, and Unknown outranks Passed in the store, so
    // recording it here would bury that proof. The diagnostic pass runs once
    // the search has given up, so its answer is the final one.
    if (
      !strategy_owns_property_table(options) ||
      options.get_bool_option("diagnose-unknown-properties"))
      goto_functionst::property_verdicts.record(
        claim.claim_key,
        property_verdictt::Unknown,
        loc,
        "inductive step could not prove this claim");
    return;
  }

  // A claim downstream of the havoc is unknowable only while the abstraction
  // admits it holding (issue #7585).
  if (
    claim.claim_after_invariant_havoc &&
    !is_loop_invariant_obligation(claim.claim_location) &&
    check_claim_unsatisfiable(local_eq) != P_UNSATISFIABLE)
  {
    weak_invariant_detected = true;
    goto_functionst::property_verdicts.record(
      claim.claim_key, property_verdictt::Unknown, loc, weak_invariant_note);
    return;
  }

  goto_functionst::property_verdicts.record(
    claim.claim_key, property_verdictt::Failed, loc);
}

void bmct::record_violated_properties(
  smt_convt &smt_conv,
  const symex_target_equationt &eq)
{
  // A subprocess SMT-LIB backend answers sat/unsat without necessarily being
  // able to produce a model (that is what --result-only buys there). Which
  // claim failed is then genuinely unknown, so leave the properties
  // NotChecked rather than guess -- and do not ask, because get-value would
  // have nothing to read.
  if (!smt_conv.has_model())
    return;

  // Symex emits a linear trace, so once the loop-invariant schema's havoc has
  // run every later claim on it is checked against the abstract state.
  bool seen_invariant_havoc = false;
  size_t claim_index = 0;

  for (const auto &step : eq.SSA_steps)
  {
    if (step.is_assignment() && step.source.pc->loop_invariant_havoc)
      seen_invariant_havoc = true;

    if (!step.is_assert() || step.ignore)
      continue;

    ++claim_index;

    // Same idiom as build_goto_trace: an unevaluatable condition renders as
    // violated, not as held.
    if (smt_conv.l_get(step.cond_expr).is_true())
      continue;

    const locationt &location = step.source.pc->location;
    const std::string description = id2string(step.comment);
    const bool abstraction_derived =
      seen_invariant_havoc && !is_loop_invariant_obligation(location);
    const bool weak_invariant =
      abstraction_derived && !invariant_refutes(eq, claim_index);
    if (weak_invariant)
      weak_invariant_detected = true;
    goto_functionst::property_verdicts.record(
      property_key(*step.source.pc, description),
      weak_invariant ? property_verdictt::Unknown : property_verdictt::Failed,
      property_location(*step.source.pc, description),
      weak_invariant ? weak_invariant_note : "");
  }
}

void bmct::error_trace(smt_convt &smt_conv, const symex_target_equationt &eq)
{
  if (options.get_bool_option("result-only"))
    return;

  log_progress("Building error trace");

  goto_tracet goto_trace;
  build_goto_trace(eq, smt_conv, goto_trace);

  std::string output_file = options.get_option("cex-output");
  if (output_file != "")
  {
    std::ofstream out(output_file);
    show_goto_trace(out, ns, goto_trace);
  }

  std::string witness_graphml_output =
    options.get_option("witness-output-graphml");
  std::string witness_yaml_output = options.get_option("witness-output-yaml");
  if (witness_graphml_output != "")
    violation_graphml_goto_trace(options, ns, goto_trace);

  if (witness_yaml_output != "")
    violation_yaml_goto_trace(options, ns, goto_trace);

  if (!options.get_option("sarif-output").empty())
  {
    sarif_goto_trace(options, ns, goto_trace, dead_store_advisories);
    dead_store_sarif_written = true;
  }

  if (options.get_bool_option("generate-testcase"))
  {
    generate_testcase_metadata();
    generate_testcase("testcase.xml", eq, smt_conv);
  }

  if (options.get_bool_option("generate-pytest-testcase"))
  {
    // Generate pytest filename based on source file: test_<module>.py
    std::string input_file = options.get_option("input-file");
    std::string module_name = pytest_generator::extract_module_name(input_file);
    std::string pytest_filename =
      pytest_generator::generate_pytest_filename(module_name);
    pytest_gen.set_values_only(options.get_bool_option("pytest-values-only"));
    pytest_gen.generate_single(
      pytest_output_dir(options), pytest_filename, eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate_single(ctest_output_dir(options), eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-foundry-testcase"))
  {
    foundry_gen.generate_single(eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-html-report"))
    generate_html_report("1", ns, goto_trace, options);

  if (options.get_bool_option("generate-json-report"))
    generate_json_report("1", ns, goto_trace);

  // A structured violation summary for tools that post-process the
  // counterexample (e.g. a source minimiser).
  const std::string violation_info_path =
    options.get_option("dump-violation-info");
  if (!violation_info_path.empty())
  {
    if (!dump_violation_info_json(violation_info_path, ns, goto_trace))
      log_warning("Failed to write violation info to {}", violation_info_path);
  }

  std::ostringstream oss;
  log_fail("\n[Counterexample]\n");
  show_goto_trace(oss, ns, goto_trace);
  log_result("{}", oss.str());
}

void bmct::generate_smt_from_equation(
  smt_convt &smt_conv,
  symex_target_equationt &eq) const
{
  std::string logic;

  if (!options.get_bool_option("int-encoding"))
  {
    logic = "bit-vector";
    logic += (!config.ansi_c.use_fixed_for_float) ? "/floating-point " : " ";
    logic += "arithmetic";
  }
  else
    logic = "integer/real arithmetic";

  log_status("Encoding remaining VCC(s) using {}", logic);

  fine_timet encode_start = current_time();
  eq.convert(smt_conv);
  fine_timet encode_stop = current_time();
  log_status(
    "Encoding to solver time: {}s", time2string(encode_stop - encode_start));
}

void bmct::keep_alive_function() const
{
  fine_timet start_time = current_time();
  while (keep_alive_running)
  {
    std::this_thread::sleep_for(std::chrono::seconds(keep_alive_interval));
    if (!keep_alive_running)
      break;

    fine_timet alive_current = current_time();
    // output runtime
    log_status(
      "Solver is still solving... Total Time: {}s",
      time2string(alive_current - start_time));
  }
}

smt_resultt bmct::run_decision_procedure(
  smt_convt &smt_conv,
  symex_target_equationt &eq) const
{
  if (options.get_bool_option("enable-keep-alive"))
  {
    keep_alive_running = true;
    keep_alive_interval =
      atoi(options.get_option("keep-alive-interval").c_str());

    if (keep_alive_interval <= 0)
      keep_alive_interval = 60; // Default interval to 60 seconds

    std::thread([this]() { keep_alive_function(); }).detach();
  }

  generate_smt_from_equation(smt_conv, eq);

  if (
    options.get_bool_option("smt-formula-too") ||
    options.get_bool_option("smt-formula-only"))
  {
    std::string smt_formula = smt_conv.dump_smt();

    // Print the SMT formula to stdout or file
    if (!smt_formula.empty())
    {
      const std::string &output_path = options.get_option("output");

      if (output_path.empty() || output_path == "-")
      {
        // Print to stdout
        fprintf(stdout, "%s", smt_formula.c_str());
      }
      else
      {
        // Print to file
        FILE *file = fopen(output_path.c_str(), "w");
        if (!file)
          log_error("Could not open output file '{}'", output_path);
        else
        {
          fprintf(file, "%s", smt_formula.c_str());
          fclose(file);
          log_status("SMT formula dumped to file: {}", output_path);
        }
      }
    }

    if (options.get_bool_option("smt-formula-only"))
      return P_SMTLIB;
  }

  log_progress("Solving with solver {}", smt_conv.solver_text());

  fine_timet sat_start = current_time();
  smt_resultt dec_result = smt_conv.dec_solve();
  fine_timet sat_stop = current_time();
  keep_alive_running = false;

  // output runtime
  log_status(
    "Runtime decision procedure: {}s", time2string(sat_stop - sat_start));

  return dec_result;
}

void bmct::report_success()
{
  // Wording deliberately avoids "unwinding assertion loop", which
  // esbmc-wrapper.py's parse_result() matches on (docs/roadmap
  // /goto-symex-verification-plan.md, R28).
  if (saw_bounded_loop_truncation)
    log_warning(
      "the unwinding bound cut a loop short while unwinding checks were "
      "disabled, so paths past the bound were assumed away rather than "
      "verified; this result holds only up to that bound");

  log_success("\nVERIFICATION SUCCESSFUL");
}

void bmct::report_failure()
{
  log_fail("\nVERIFICATION FAILED");
}

void bmct::report_unknown()
{
  log_fail("\nVERIFICATION UNKNOWN");
}

void bmct::report_violation()
{
  // When every violated claim sits downstream of a --loop-invariant-check
  // havoc, no counterexample witnesses a state the program can reach. An
  // over-approximation can prove, never refute: the invariant being too weak
  // is "cannot prove", not "the program is wrong" (issue #7480).
  if (
    !weak_invariant_detected ||
    goto_functionst::property_verdicts.has_violation())
  {
    report_failure();
    return;
  }

  log_warning(
    "every violated claim lies downstream of a loop invariant havoc, so its "
    "counterexample is against the abstraction rather than the program; "
    "strengthen the invariant to decide the claim");
  report_unknown();
  verdict_is_unknown = true;
}

/// UNSAT iff no feasible path satisfies the kept claim. A violation found
/// downstream of a loop-invariant havoc is then one the invariant itself
/// forces, not an artefact of the abstraction (issue #7585).
smt_resultt
bmct::check_claim_unsatisfiable(symex_target_equationt &local_eq) const
{
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  local_eq.convert(
    *solver, symex_target_equationt::assertion_modet::Satisfiable);
  return solver->dec_solve();
}

/// Whether the invariant leaves the claim at \p claim_index no way to hold.
/// The single-formula path has no per-claim equation, so slice one the way
/// multi_property_check does and run the same probe.
bool bmct::invariant_refutes(
  const symex_target_equationt &eq,
  size_t claim_index)
{
  symex_target_equationt local_eq = eq;
  claim_slicer claim(claim_index, false, false, ns);
  claim.run(local_eq.SSA_steps);
  return check_claim_unsatisfiable(local_eq) == P_UNSATISFIABLE;
}

smt_resultt bmct::check_vacuity(symex_target_equationt &local_eq) const
{
  // Re-encode in vacuity mode: each kept assertion contributes its path
  // assumption to the OR'd disjunction instead of `not(assumpt -> claim)`.
  // The result is UNSAT iff the path to every kept claim is unreachable.
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  local_eq.convert(
    *solver, symex_target_equationt::assertion_modet::PathReachable);
  return solver->dec_solve();
}

// True when a discharged claim is a candidate for vacuity probing. Vacuity
// asks whether a claim held only because its path was dead, which is a
// question about what the user meant to state -- so the probe is limited to
// claims the user wrote. An auto-generated safety check (overflow, array
// bounds, ...) discharged on an unreachable failure path is the intended
// result, not a warning, and every correct program with bounded arithmetic
// produces some (#5327). Naming the admitted claims rather than the rejected
// ones also keeps a newly added built-in check from poisoning verdicts.
// Excluded for a second reason: the loop-invariant pass's own synthetic
// assertions sit under an ASSUME(false) terminator, so any claim after the
// first loop's inductive step would always probe vacuous.
static bool is_vacuity_probe_candidate(const std::string &claim_property)
{
  return claim_property == "assertion" ||
         claim_property == "contract ensures" ||
         claim_property == "assigns compliance";
}

void bmct::show_program(const symex_target_equationt &eq)
{
  unsigned int count = 1;
  std::ostringstream oss;
  if (config.options.get_bool_option("ssa-symbol-table"))
    ::show_symbol_table_plain(ns, oss);

  languagest languages(ns, configured_language());

  oss << "\nProgram constraints: \n";

  bool sliced = config.options.get_bool_option("ssa-sliced");

  for (auto const &it : eq.SSA_steps)
  {
    if (!(it.is_assert() || it.is_assignment() || it.is_assume()))
      continue;

    if (it.ignore && !sliced)
      continue;

    oss << "// " << it.source.pc->location_number << " ";
    oss << it.source.pc->location.as_string();
    if (!it.comment.empty())
      oss << " (" << it.comment << ")";
    oss << "\n/* " << count << " */ ";

    std::string string_value;
    languages.from_expr(migrate_expr_back(it.cond), string_value);

    if (it.is_assignment())
    {
      oss << string_value << "\n";
    }
    else if (it.is_assert())
    {
      oss << "(assert)" << string_value << "\n";
    }
    else if (it.is_assume())
    {
      oss << "(assume)" << string_value << "\n";
    }
    else if (it.is_renumber())
    {
      oss << "renumber: " << from_expr(ns, "", it.lhs) << "\n";
    }

    if (!migrate_expr_back(it.guard).is_true())
    {
      languages.from_expr(migrate_expr_back(it.guard), string_value);
      oss << std::string(i2string(count).size() + 3, ' ');
      oss << "guard: " << string_value << "\n";
    }

    oss << '\n';
    count++;
  }
  log_status("{}", oss.str());
}

void bmct::report_trace(smt_resultt &res, const symex_target_equationt &eq)
{
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");
  bool show_cex = options.get_bool_option("show-cex");

  switch (res)
  {
  case P_UNSATISFIABLE:
    if (is && term)
    {
    }
    else if (!bs)
    {
      successful_trace(eq);
    }
    break;

  case P_SATISFIABLE:
    // A verdict can be reached without a solver having been kept — no model to
    // read, so there is no trace to build and dereferencing it would crash.
    if (!runtime_solver)
      break;
    // An inductive-step or forward-condition model starts from a havoc'd
    // state, so it witnesses no violation of the program and must not reach a
    // verdict (multi_property_check draws the same line at its
    // `is ? Unknown : Failed`). Printing that trace under --show-cex is a
    // separate decision from recording it, so the two conditions are separate.
    if (!is && !fc)
      record_violated_properties(*runtime_solver, eq);
    if ((!bs && show_cex) || (!is && !fc))
      error_trace(*runtime_solver, eq);
    break;

  default:
    break;
  }
}

/*
  For incremental-bmc and k-induction
  Whenever an error_trace or successful_trace is reported
  we finish reasoning this claims, thereby converting it to SKIP
*/
void bmct::clear_verified_claims_in_ssa(
  symex_target_equationt &local_eq,
  const claim_slicer &claim,
  const bool &is_goto_cov)
{
  for (auto &step : local_eq.SSA_steps)
  {
    if (!step.is_assert())
      continue;

    if (!step.source.is_set)
      continue;

    bool loc_match = (step.source.pc->location.as_string() == claim.claim_loc);
    bool expr_match = false;

    if (is_goto_cov)
      expr_match =
        (step.source.pc->location.comment().as_string() == claim.claim_msg);
    else
      expr_match = (from_expr(ns, "", step.guard) == claim.claim_msg);

    if (loc_match && expr_match)
    {
      step.cond = step.cond = gen_true_expr();
    }
  }
}

void bmct::clear_verified_claims_in_goto(
  const claim_slicer &claim,
  const bool &is_goto_cov)
{
  std::lock_guard lock(goto_functionst::clear_claims_mutex);
  for (auto &func : symex->goto_functions.function_map)
  {
    for (auto &instr : func.second.body.instructions)
    {
      if (!instr.is_assert())
        continue;

      // Only the instruction's own assertion: --loop-invariant copies a loop
      // body, and a check symex raises while evaluating an assertion's guard
      // shares its instruction, so neither may skip it on its own verdict.
      const bool match =
        is_goto_cov
          ? instr.location.as_string() == claim.claim_loc &&
              instr.location.comment().as_string() == claim.claim_msg
          : &instr == claim.claim_instruction &&
              goto_symext::assertion_message(ns, instr) == claim.claim_comment;

      if (match)
      {
        instr.make_skip();
      }
    }
  }
}

namespace
{
/// Run one claim's job, mapping a thrown diagnostic to ERROR. An exception
/// escaping a thread's entry function is std::terminate, with none of the
/// handling bmct::run_thread gives the sequential path, so the parallel
/// scheduler has to convert it here.
template <typename jobt>
void run_job_guarded(
  const jobt &job,
  const size_t &i,
  std::mutex &result_mutex,
  smt_resultt &final_result,
  std::atomic<bool> &report_incomplete)
{
  try
  {
    job(i);
  }
  catch (const std::string &error_str)
  {
    log_error("{}", error_str);
    report_incomplete = true;
    std::lock_guard lock(result_mutex);
    if (final_result != P_SATISFIABLE)
      final_result = P_ERROR;
  }
}

/// True when the multi-witness report must avoid box-drawing glyphs. A console
/// that is not reading UTF-8 renders them as mojibake on every line of the
/// report (esbmc/esbmc#4311). On Windows the console's code page is queried;
/// on POSIX the locale environment is read. An unset locale is treated as
/// UTF-8 so the common CI shape keeps the richer output.
bool ascii_report(const optionst &options)
{
  if (options.get_bool_option("ascii-report"))
    return true;
#ifdef _WIN32
  // A Windows console is cp1252 by default but can be switched to UTF-8
  // (`chcp 65001`), so ask it rather than assuming: assuming would also cost
  // the richer output on a console that renders it correctly.
  return GetConsoleOutputCP() != CP_UTF8;
#else
  for (const char *var : {"LC_ALL", "LC_CTYPE", "LANG"})
  {
    const char *val = std::getenv(var);
    if (!val || !*val)
      continue;
    std::string v(val);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    return v.find("utf-8") == std::string::npos &&
           v.find("utf8") == std::string::npos;
  }
  return false;
#endif
}

/// States nearest the failure are the ones that explain it; the rest is
/// prologue repeated almost verbatim across every witness (#4311). Keep the
/// last @p keep of them and replace what precedes with a count, so the reader
/// still knows the trace was shortened. Operates on the rendered text because
/// that is what "states in the report" means -- show_goto_trace decides for
/// itself which steps become states.
std::string keep_last_trace_states(const std::string &rendered, size_t keep)
{
  // Rendered states start at column 0 with "State ".
  std::vector<size_t> starts;
  for (size_t pos = 0; pos != std::string::npos;)
  {
    size_t hit = rendered.compare(pos, 6, "State ") == 0
                   ? pos
                   : rendered.find("\nState ", pos);
    if (hit == std::string::npos)
      break;
    if (rendered.compare(hit, 6, "State ") != 0)
      ++hit; // skip the newline the search matched on
    starts.push_back(hit);
    pos = hit + 6;
  }

  if (starts.size() <= keep)
    return rendered;

  const size_t omitted = starts.size() - keep;
  const size_t cut = starts[omitted];
  return rendered.substr(0, starts.front()) + "... " + std::to_string(omitted) +
         " earlier states omitted (--full-traces to show them) ...\n\n" +
         rendered.substr(cut);
}

/// Matches the K=50 the issue proposes; large enough to keep the explanatory
/// tail of a trace, small enough that N witnesses stay readable.
constexpr size_t kMaxReportedStates = 50;
} // namespace

void bmct::report_multi_property_trace(
  const smt_resultt &res,
  const std::vector<witness_recordt> &witnesses,
  enumeration_stop_reasont stop_reason,
  const std::string &msg,
  bool reachability_trace)
{
  if (options.get_bool_option("result-only"))
    return;

  switch (res)
  {
  case P_UNSATISFIABLE:
    log_success("Claim '{}' holds up to the current K", msg);
    return;

  case P_SATISFIABLE:
    break;

  default:
    log_fail("Claim '{}' could not be solved", msg);
    return;
  }

  // Single-witness textual output: keep the existing "[Counterexample]" form.
  // This preserves the look of every existing failing test in regression/.
  // Skip this path when --all-witnesses was requested (stop_reason != Disabled)
  // so the structured footer is still emitted at N=1 — otherwise a CapHit
  // with cap=1 would silently look identical to a real "only-one-witness"
  // result.
  if (
    witnesses.size() <= 1 && stop_reason == enumeration_stop_reasont::Disabled)
  {
    std::ostringstream oss;
    if (reachability_trace)
      log_success("\n[Reachability trace]\n");
    else
      log_fail("\n[Counterexample]\n");
    if (!witnesses.empty())
      show_goto_trace(oss, ns, witnesses.front().trace, reachability_trace);
    log_result("{}", oss.str());
    return;
  }

  // Multi-witness rendering: structured per-witness blocks, then a footer.
  // Goal: highlight the *inputs* (the part that varies across witnesses) and
  // avoid dumping N copies of nearly-identical traces. The full trace for
  // each witness is still emitted unless --compact-trace is on, but they
  // are clearly separated and labelled.
  std::ostringstream oss;
  // ASCII-only header: en-dash and similar non-ASCII glyphs get
  // mojibake'd on Windows' default cp1252 console, breaking regression
  // matching and reader output. The box-drawing glyphs further down
  // are cosmetic and only appear at N>1; ASCII-fallback there is
  // tracked separately (#4311).
  oss << (reachability_trace ? "\n[Reachability traces - "
                             : "\n[Counterexamples - ")
      << witnesses.size() << " witnesses";
  // An incremental run already enumerates at every failing k, not just the
  // first, so without the bound a reader cannot tell which unwinding produced
  // a block -- or that two blocks are different unwindings rather than a
  // repeat (esbmc/esbmc#4314). Plain BMC has one bound, where this is noise.
  if (options.get_bool_option("incremental-bmc"))
  {
    const std::string k = options.get_option("unwind");
    oss << " at k = " << (k.empty() ? "0" : k);
  }
  oss << "]\n\n";
  // Say up front that this is a truncated enumeration. The same fact reaches
  // the Summary footer below, but that sits after every witness block -- tens
  // of kilobytes on a real program -- so a reader can easily act on a partial
  // list without realising it (#4311).
  if (stop_reason == enumeration_stop_reasont::CapHit)
    oss << "  NOTE: --max-witnesses cap reached; more witnesses may exist.\n\n";
  // The inputs are the part that actually differs between witnesses, and they
  // are what a reader needs first. Per-witness they sit one trace apart, so on
  // a real program comparing them means paging through tens of kilobytes of
  // near-identical trace. Collect them up front (#4311). ASCII only, for the
  // same cp1252 reason as the header above.
  {
    bool any_inputs = false;
    for (const witness_recordt &w : witnesses)
      if (!w.nondet_inputs.empty())
      {
        any_inputs = true;
        break;
      }

    if (any_inputs)
    {
      oss << "  Inputs by witness:\n";
      for (size_t i = 0; i < witnesses.size(); ++i)
      {
        const witness_recordt &w = witnesses[i];
        oss << "    #" << (i + 1) << " : ";
        if (w.nondet_inputs.empty())
          oss << "(none)";
        else
          for (size_t k = 0; k < w.nondet_inputs.size(); ++k)
          {
            if (k)
              oss << ", ";
            oss << "[" << k << "] = "
                << from_expr(
                     ns,
                     "",
                     w.nondet_inputs[k].value_expr,
                     presentationt::WITNESS);
          }
        oss << "\n";
      }
      oss << "\n";
    }
  }
  // Box-drawing glyphs are mojibake'd by a console that is not reading UTF-8
  // -- Windows' default cp1252 above all -- which at N witnesses corrupts
  // every line of the report (esbmc/esbmc#4311).
  const bool ascii = ascii_report(options);
  const std::string bar = ascii ? "|" : "│";
  const std::string head_open = ascii ? "  +- " : "  ┌─ ";
  const std::string head_fill =
    ascii ? " -----------------------------" : " ─────────────────────────────";
  const std::string foot =
    ascii ? "  +---------------------------------------------\n\n"
          : "  └──────────────────────────────────────────────\n\n";

  for (size_t i = 0; i < witnesses.size(); ++i)
  {
    const witness_recordt &w = witnesses[i];
    oss << head_open << "Witness " << (i + 1) << " of " << witnesses.size()
        << head_fill << "\n";
    oss << "  " << bar << "  Inputs : ";
    if (w.nondet_inputs.empty())
    {
      oss << "(none)\n";
    }
    else
    {
      for (size_t k = 0; k < w.nondet_inputs.size(); ++k)
      {
        if (k)
          oss << ", ";
        // Use WITNESS presentation to render bit-distinct floats with
        // round-trippable precision; otherwise the default HUMAN flags
        // collapse e.g. several near-MAX floats to the same string.
        oss << "[" << k << "] = "
            << from_expr(
                 ns, "", w.nondet_inputs[k].value_expr, presentationt::WITNESS);
      }
      oss << "\n";
    }
    oss << "  " << bar << "  Trace  :\n";
    {
      std::ostringstream tr;
      show_goto_trace(tr, ns, w.trace, reachability_trace);
      // Indent the trace under the box.
      std::string s = tr.str();
      if (!options.get_bool_option("full-traces"))
        s = keep_last_trace_states(s, kMaxReportedStates);
      std::string indented;
      indented.reserve(s.size() + 8);
      const std::string lead = "  " + bar + "    ";
      indented += lead;
      for (char c : s)
      {
        indented += c;
        if (c == '\n')
          indented += lead;
      }
      oss << indented << "\n";
    }
    oss << foot;
  }

  oss << "Summary: " << witnesses.size()
      << (reachability_trace ? " distinct input tuples reach this goal"
                             : " distinct input tuples violate this property")
      << " (enumeration stopped: ";
  switch (stop_reason)
  {
  case enumeration_stop_reasont::Unsat:
    oss << "UNSAT after " << witnesses.size() << " witnesses";
    break;
  case enumeration_stop_reasont::CapHit:
    oss << "--max-witnesses cap reached";
    break;
  case enumeration_stop_reasont::NoInputs:
    oss << "no enumerable nondet inputs — more witnesses may exist";
    break;
  case enumeration_stop_reasont::Error:
    oss << "solver returned error/unknown — more witnesses may exist";
    break;
  case enumeration_stop_reasont::Disabled:
    oss << "single-witness mode";
    break;
  }
  oss << ")\n";

  if (reachability_trace)
    log_success("\n[Reachability]\n");
  else
    log_fail("\n[Counterexample]\n");
  log_result("{}", oss.str());
}

// Prettify C-level expression strings for Solidity coverage reports.
// Strips C casts, maps internal names to Solidity names, etc.
static std::string prettify_solidity_expr(const std::string &expr)
{
  if (config.language.lid != language_idt::SOLIDITY)
    return expr;

  std::string s = expr;

  // Remove C-style casts: (signed int), (unsigned int), (signed long int), etc.
  // Also handles _ExtInt(N) casts like (unsigned _ExtInt(256))
  static const std::regex cast_re(
    R"(\((?:signed|unsigned)\s+(?:_ExtInt\(\d+\)|(?:long\s+)?(?:long\s+)?int)\))");
  s = std::regex_replace(s, cast_re, "");

  // Remove this-> prefix (Solidity state variables)
  static const std::regex this_re(R"(this->)");
  s = std::regex_replace(s, this_re, "");

  // Map internal Solidity global variable names to their Solidity equivalents
  static const std::vector<std::pair<std::regex, std::string>> name_map = {
    {std::regex(R"(\bmsg_sender\b)"), "msg.sender"},
    {std::regex(R"(\bmsg_value\b)"), "msg.value"},
    {std::regex(R"(\bmsg_sig\b)"), "msg.sig"},
    {std::regex(R"(\bmsg_data\b)"), "msg.data"},
    {std::regex(R"(\btx_origin\b)"), "tx.origin"},
    {std::regex(R"(\btx_gasprice\b)"), "tx.gasprice"},
    {std::regex(R"(\bblock_number\b)"), "block.number"},
    {std::regex(R"(\bblock_timestamp\b)"), "block.timestamp"},
    {std::regex(R"(\bblock_coinbase\b)"), "block.coinbase"},
    {std::regex(R"(\bblock_difficulty\b)"), "block.difficulty"},
    {std::regex(R"(\bblock_gaslimit\b)"), "block.gaslimit"},
    {std::regex(R"(\bblock_chainid\b)"), "block.chainid"},
    {std::regex(R"(\bblock_basefee\b)"), "block.basefee"},
    {std::regex(R"(\bblock_blobbasefee\b)"), "block.blobbasefee"},
    {std::regex(R"(\bblock_prevrandao\b)"), "block.prevrandao"},
  };
  for (const auto &[re, repl] : name_map)
    s = std::regex_replace(s, re, repl);

  // Remove redundant parentheses left by cast removal, e.g. ((x)) -> (x)
  // Iteratively reduce until stable (handles nested cases)
  static const std::regex double_paren(R"(\((\([^()]*\))\))");
  std::string prev;
  do
  {
    prev = s;
    s = std::regex_replace(s, double_paren, "$1");
  } while (s != prev);

  // Remove parens in array index: [(...)] -> [...]
  static const std::regex bracket_paren(R"(\[\(([^()]*)\)\])");
  s = std::regex_replace(s, bracket_paren, "[$1]");

  // Prettify Solidity internal symbol IDs: sol:@C@Contract@F@func#N -> func
  // Appears in "function entry: sol:@C@..." messages
  static const std::regex sol_id_re(R"(sol:@C@\w+@F@(\w+)#\d*)");
  s = std::regex_replace(s, sol_id_re, "$1");

  // Clean up extra spaces from removed casts
  static const std::regex multi_space(R"(  +)");
  s = std::regex_replace(s, multi_space, " ");

  // Remove leading/trailing whitespace
  auto start = s.find_first_not_of(' ');
  auto end = s.find_last_not_of(' ');
  if (start != std::string::npos)
    s = s.substr(start, end - start + 1);

  return s;
}

// Solidity complete-path coverage: can THIS run establish that a path is
// UNREACHABLE (status I), as opposed to merely "no witness found here" (U)?
//
// It could only do so if the exploration over-approximated every reachable
// state. Nothing in Solidity coverage mode does:
//
//  * `--solidity-max-tx N` with N > 0 emits N straight-line transactions —
//    bounded by construction.
//  * `--solidity-max-tx 0` asks the frontend for the `while (nondet_bool())
//    dispatch()` driver, which READS as unbounded — but that loop is then
//    destroyed. esbmc_parseoptions.cpp::process_goto_program walks every
//    function whose name contains `_ESBMC_Main` and calls make_skip() on each
//    `is_backwards_goto()`, under a condition (`is_coverage &&
//    !--coverage-multi-tx`) whose `is_coverage` disjunction includes
//    `--solidity-path-coverage`. One guarded transaction is left, so max_tx 0
//    is the SHALLOWEST setting, not an unbounded one.
//
//    Verified three independent ways, not taken from documentation:
//      - the code above;
//      - `--show-loops` on the same contract lists `_ESBMC_Main_S` as a loop
//        WITHOUT the coverage flag and does not list it WITH it (every other
//        loop identical);
//      - behaviour: on `arm(); fire();`, where fire()'s guarded path needs a
//        prior transaction, that path is REACHED at --solidity-max-tx 2 and
//        NOT reached at --solidity-max-tx 0.
//  * the entry state is whatever the constructor left; state variables are
//    never havoc'd, so an UNSAT only says "not reachable from THIS entry
//    state", not "not reachable".
//
// Treating max_tx == 0 as unbounded — which this code used to do — would dress
// a one-transaction budget up as a proof of unreachability, the exact failure
// the tri-state exists to prevent. So every non-refuted path is U, flagged
// `bounded_holds` when it held at this exploration.
//
// Kept as a function rather than deleted so that the day a havoc'd-entry or
// loop-live exploration mode exists, this is the single place to change.
static bool path_cov_can_prove_unreachable()
{
  return false;
}

// Parse location string "file X line Y column Z function F" into components
static nlohmann::json parse_claim_location(const std::string &loc)
{
  nlohmann::json j;
  j["file"] = "";
  j["line"] = 0;
  j["column"] = 0;
  j["function"] = "";

  std::vector<std::string> words;
  {
    std::istringstream iss(loc);
    std::string w;
    while (iss >> w)
      words.push_back(std::move(w));
  }

  auto is_key = [](const std::string &w) {
    return w == "file" || w == "line" || w == "column" || w == "function";
  };

  for (size_t i = 0; i < words.size();)
  {
    const std::string key = words[i++];
    std::string val;
    while (i < words.size() && !is_key(words[i]))
    {
      if (!val.empty())
        val += " ";
      val += words[i++];
    }
    if (key == "line" || key == "column")
      j[key] = atoi(val.c_str());
    else if (key == "file" || key == "function")
      j[key] = val;
  }
  return j;
}

// Predicate used by both the final and the verbose k-path reporters.
static bool is_kpath_maximal(const std::string &claim_sig)
{
  const auto &redundant = goto_coveraget::k_path_spanning_redundant;
  // claim_sig = "msg\tloc"; loc has no tabs, so rfind is robust if a
  // future emission path puts a tab in msg.
  const auto tab = claim_sig.rfind('\t');
  return redundant.count(
           {claim_sig.substr(0, tab), claim_sig.substr(tab + 1)}) == 0;
}

// Advisory dead-code reporter for --dead-code-check (CWE-561, issue #4495).
//
// Reuses the branch-coverage instrumentation: a probe `assert(c)` (an
// instrumented assertion over a branch guard) that multi_property_check never
// violated proves `!c` infeasible up to the current unwinding bound — so `!c`
// is the dead direction, and the advisory names goto_coveraget::claim_negation
// rather than the claim's own comment. claim_negation is keyed by, and rebuilt
// with, goto_coveraget::all_claims, so the dead set is exactly the entries
// below that reached_claims does not hold. Findings are advisory: they are
// printed as a separate [Dead code] section and, when --sarif-output is set,
// emitted at SARIF note level. They never flip the verdict (see report_result).
static void report_dead_code(
  const optionst &options,
  const std::unordered_set<std::string> &reached_claims,
  const std::vector<dead_store_advisoryt> &dead_stores)
{
  std::vector<dead_code_finding_t> findings;

  for (const auto &[claim, dead_guard] : goto_coveraget::claim_negation)
  {
    const auto &[comment, loc] = claim;
    if (reached_claims.count(comment + "\t" + loc))
      continue; // reachable branch direction — live code

    nlohmann::json parsed = parse_claim_location(loc);
    dead_code_finding_t f;
    f.file = parsed["file"].get<std::string>();
    f.line = static_cast<unsigned>(parsed["line"].get<int>());
    f.message = dead_guard.empty()
                  ? "dead code: unreachable branch"
                  : "dead code: unreachable branch [guard: " + dead_guard + "]";
    findings.push_back(std::move(f));
  }

  log_success("\n[Dead code]\n");
  if (findings.empty())
    log_result("No provably-dead code found.");
  else
  {
    // Soundness is bounded by the unwinding depth, like every BMC result: a
    // branch reachable only beyond the explored bound is reported here too.
    // Scope the advisory accordingly so it is not read as an absolute proof
    // (increase --unwind for programs with loops).
    log_status(
      "The following branches are unreachable up to the current unwinding "
      "bound:");

    const std::string cwes = format_cwe_list(dead_code_cwe_rule().cwes);
    for (const auto &f : findings)
    {
      if (f.line > 0)
        log_result("{}:{}: {}", f.file, f.line, f.message);
      else
        log_result("{}", f.message);
      log_result("  CWE: {}", cwes);
    }
  }

  // Mirror the findings into SARIF when requested. A clean run still emits a
  // well-formed document with an empty results array, so --sarif-output never
  // yields a missing file (issue #4495). Dead-store advisories go into the same
  // document: they share the one output path, so a second write would truncate
  // these findings away.
  sarif_dead_code(options, findings, dead_stores);
}

/// Coverage numerator over the branch instrumentation. reached_claims records
/// every claim symex refuted, including ones outside the instrumentation (an
/// uncaught exception, say), so its raw size can exceed the goal count and
/// report over 100% (#7296). all_claims is the instrumented set.
static size_t
count_reached_goals(const std::unordered_set<std::string> &reached_claims)
{
  size_t reached = 0;
  for (const auto &[comment, loc] : goto_coveraget::all_claims)
    if (reached_claims.count(comment + "\t" + loc))
      ++reached;
  return reached;
}

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims,
  pytest_generator &pytest_gen,
  ctest_generator &ctest_gen,
  foundry_generator &foundry_gen,
  const std::string &partial_reason)
{
  // --dead-code-check reuses the coverage machinery for instrumentation but
  // reports its results as CWE-561 advisories rather than a coverage summary.
  //
  // The advisory is *not* emitted here: report_coverage runs inside
  // multi_property_check, i.e. once per thread interleaving, so a branch
  // reachable only under a later ordering would be called dead on the strength
  // of the first interleaving alone. bmct::start_bmc emits it once, after
  // exploration finishes and probe reachability has accumulated across every
  // interleaving (issue #4495).
  if (options.get_bool_option("dead-code-check"))
    return;

  // Published to the U-reason classifier before anything reads it, so a claim
  // that never got a verdict is filed `run-died-before-solving` rather than
  // `not-solved-this-run` (which means "the simplifier folded it away", a fact
  // about the claim and not about the run), and so audit_entry_liveness stops
  // treating an un-entered unit as a defect. Both consumers run inside this
  // function, below.
  const bool is_partial = !partial_reason.empty();
  if (is_partial)
    goto_coveraget::path_cov_partial_reason = partial_reason;

  bool is_assert_cov = options.get_bool_option("assertion-coverage") ||
                       options.get_bool_option("assertion-coverage-claims");
  bool is_cond_cov = options.get_bool_option("condition-coverage") ||
                     options.get_bool_option("condition-coverage-claims") ||
                     options.get_bool_option("condition-coverage-rm") ||
                     options.get_bool_option("condition-coverage-claims-rm");
  bool is_branch_cov = options.get_bool_option("branch-coverage") ||
                       options.get_bool_option("branch-coverage-claims");
  bool is_branch_func_cov =
    options.get_bool_option("branch-function-coverage") ||
    options.get_bool_option("branch-function-coverage-claims");
  // `k-path-coverage` itself stores the CLI integer N; the dedicated
  // boolean enable flag is set by parseoptions when either CLI flag is
  // present. This avoids `get_bool_option("k-path-coverage")` returning 0
  // (false) for valid invocations like `--k-path-coverage` (no value) or
  // `--k-path-coverage=0` (rejected at parse time, but defensive here).
  bool is_k_path_cov = options.get_bool_option("k-path-coverage-enabled");
  // Solidity complete-path coverage: one goal per enumerated entry->exit
  // path, keyed by the "fn:path:enc" comment (see solidity_path_coverage()).
  bool is_path_cov = options.get_bool_option("solidity-path-coverage-enabled");

  // Truncation disclosure. With --no-unwinding-assertions (which coverage
  // mode turns on automatically whenever --unwind is given, and which the
  // k-induction base/inductive phases set too) a loop that reaches its
  // bound is cut with an ASSUME rather than flagged with an assertion.
  // Every path that needed one more iteration disappears, so the code
  // after the loop can become unreachable and its goals are reported as
  // uncovered -- with no diagnostic at all, next to "VERIFICATION
  // SUCCESSFUL". Say so, and name the loops, so a 0% report is not read as
  // "the harness explored everything and covered nothing".
  {
    std::lock_guard lk(goto_functionst::truncated_loops_mutex);
    if (
      !goto_functionst::truncated_loops.empty() &&
      options.get_bool_option("no-unwinding-assertions"))
    {
      log_warning(
        "Coverage may be UNDER-REPORTED: {} loop(s) hit the unwind bound "
        "while --no-unwinding-assertions was active, so the paths that "
        "needed more iterations were silently assumed away. Goals reachable "
        "only through those paths are counted as uncovered. Raise --unwind, "
        "use --unwindset/--unwindsetname for the specific loop, or switch to "
        "--k-induction / --incremental-bmc. Loops truncated:",
        goto_functionst::truncated_loops.size());
      for (const auto &l : goto_functionst::truncated_loops)
        log_warning("  {}", l);
    }
  }

  if (is_assert_cov)
  {
    const int total = goto_coveraget::total_assert;
    const int tracked_instance = reached_mul_claims.size();
    const int total_instance = goto_coveraget::total_assert_ins;

    // Assertions never observed during symbolic execution: either their
    // branch was pruned by a constant guard, or their path guard is
    // unsatisfiable so the claim is vacuously valid (discussion #5745).
    std::vector<std::string> unreached_claims;
    for (const auto &[claim_msg, claim_loc] : goto_coveraget::all_claims)
    {
      const std::string claim_sig = claim_msg + "\t" + claim_loc;
      if (reached_mul_claims.count(claim_sig) == 0)
        unreached_claims.push_back(claim_sig);
    }

    log_success("\n[Coverage]\n");
    // The total assertion instances include the assert inside the source file,
    // the unwinding asserts, the claims inserted during the goto-check and so
    // on. "Total/Unreached Asserts" count static claims; the "Instances" lines
    // count their goto-unwound copies.
    log_result("Total Asserts: {}", total);
    log_result("Unreached Asserts: {}", unreached_claims.size());
    if (total_instance >= tracked_instance)
      log_result("Total Assertion Instances: {}", total_instance);
    else
    {
      // this could be
      // 1. the loop is too large that we cannot goto-unwind it
      // 2. the loop is somewhat non-deterministic that we cannot run
      // goto-unwind
      log_result("Total Assertion Instances: unknown / non-deterministic");
      note_cov_incomplete(
        "the total number of assertion instances could not be determined "
        "(a loop bound is non-deterministic or too large to unwind)");
    }
    log_result("Reached Assertion Instances: {}", tracked_instance);

    // show claims
    if (options.get_bool_option("assertion-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_mul_claims)
      {
        log_status("  {} : REACHED", prettify_solidity_expr(claim));
      }
      // unreached claims:
      for (const auto &claim : unreached_claims)
      {
        log_status("  {} : UNREACHED", prettify_solidity_expr(claim));
      }
    }

    if (total_instance != 0)
    {
      if (total_instance >= tracked_instance)
        log_result(
          "Assertion Instances Coverage: {}%",
          tracked_instance * 100.0 / total_instance);
      else
        log_result("Assertion Instances Coverage Unknown");
    }
    else
      log_result("Assertion Instances Coverage: 0%");
  }

  else if (is_cond_cov)
  {
    log_success("\n[Coverage]\n");

    // not all the claims are cond-cov instrumentations
    // thus we need to skip the irrelevant claims like unwinding assertions
    // when comparing 'total_cond_assert' and 'reached_claims'
    const std::set<std::pair<std::string, std::string>> &total_cond_assert =
      goto_coveraget::total_cond;
    const size_t total_instance = total_cond_assert.size();
    size_t reached_instance = 0;
    size_t short_circuit_instance = 0;
    size_t sat_instance = 0;
    size_t unsat_instance = 0;

    // show claims
    bool cond_show_claims =
      options.get_bool_option("condition-coverage-claims") ||
      options.get_bool_option("condition-coverage-claims-rm");

    // Local copy: the JSON writer downstream reads the original
    // reached_claims; mutating it here would mark every claim uncovered.
    auto reached_claims_local = reached_claims;
    auto total_cond_assert_cpy = total_cond_assert;
    for (const auto &claim_pair : total_cond_assert)
    {
      std::string claim_msg = claim_pair.first;
      std::string claim_loc = claim_pair.second;
      std::string claim_sig = claim_msg + "\t" + claim_loc;
      if (reached_claims_local.count(claim_sig))
      {
        // show sat claims
        if (cond_show_claims)
          log_status("  {} : SATISFIED", prettify_solidity_expr(claim_sig));

        // update counter +=2
        // as we handle ass and !ass at the same time
        reached_instance += 2;

        // update sat counter
        ++sat_instance;

        // prevent double count
        reached_claims_local.erase(claim_sig);
        total_cond_assert_cpy.erase(claim_pair);

        // reversal: obtain !ass
        if (
          claim_msg[0] == '!' && claim_msg[1] == '(' && claim_msg.back() == ')')
          // e.g. !(a==1)
          claim_msg = claim_msg.substr(2, claim_msg.length() - 3);
        else
          claim_msg = "!(" + claim_msg + ")";
        std::string r_claim_sig = claim_msg + "\t" + claim_loc;

        if (reached_claims_local.count(r_claim_sig))
        {
          ++sat_instance;
          if (cond_show_claims)
            log_result("  {} : SATISFIED", prettify_solidity_expr(r_claim_sig));
        }
        else
        {
          ++unsat_instance;
          if (cond_show_claims)
            log_result(
              "  {} : UNSATISFIED", prettify_solidity_expr(r_claim_sig));
        }

        // prevent double count
        // e.g if( a ==0 && a == 0)
        // we only count a==0 and !(a==0) once
        reached_claims_local.erase(r_claim_sig);
        std::pair<std::string, std::string> _pair =
          std::make_pair(claim_msg, claim_loc);
        total_cond_assert_cpy.erase(_pair);
      }
    }

    // the remain unreached instrumentations are regarded as short-circuited
    //! the reached_claims might not be empty (due to unwinding assertions)
    short_circuit_instance = total_cond_assert_cpy.size();

    // show short-circuited:
    if (cond_show_claims && short_circuit_instance > 0)
    {
      log_success("[Short Circuited Conditions]\n");
      for (const auto &claim_pair : total_cond_assert_cpy)
      {
        std::string claim_msg = claim_pair.first;
        std::string claim_loc = claim_pair.second;
        std::string claim_sig = claim_msg + "\t" + claim_loc;
        log_result("  {}", prettify_solidity_expr(claim_sig));
      }
    }

    // show the number
    log_result("Reached Conditions:  {}", reached_instance);
    log_result("Short Circuited Conditions:  {}", short_circuit_instance);
    log_result(
      "Total Conditions:  {}\n", reached_instance + short_circuit_instance);

    log_result("Condition Properties - SATISFIED:  {}", sat_instance);
    log_result("Condition Properties - UNSATISFIED:  {}\n", unsat_instance);

    if (total_instance != 0)
      log_result(
        "Condition Coverage: {}%", sat_instance * 100.0 / total_instance);
    else
      log_result("Condition Coverage: 0%");
  }

  else if (is_branch_cov)
  {
    const size_t total = goto_coveraget::total_branch;
    const bool cov_set_active = !goto_coveraget::covered_set_outpath.empty();

    // With the cross-run covered-set, the numerator is the universe edges
    // either witnessed this run or already persisted.
    size_t tracked_instance = count_reached_goals(reached_claims);
    if (cov_set_active)
    {
      tracked_instance = 0;
      for (const auto &[cond, loc] : goto_coveraget::all_claims)
        if (
          goto_coveraget::covered_set.count({cond, loc}) ||
          reached_claims.count(cond + "\t" + loc))
          ++tracked_instance;
    }
    log_success("\n[Coverage]\n");
    log_result("Branches : {}", total);
    log_result("Reached : {}", tracked_instance);

    // show claims
    if (options.get_bool_option("branch-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_claims)
        log_status("  {}", prettify_solidity_expr(claim));
    }

    if (total != 0)
      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    else
      log_result("Branch Coverage: N/A (no branches)");

    // Re-sync the signal-safe snapshot to the AUTHORITATIVE number:
    // this is the only place reached_claims is erased (short-circuit /
    // non-universe pruning above), and tracked_instance is exactly the
    // mode-correct numerator just printed. Storing it into the active
    // counter makes a subsequent kill (e.g. inductive step after the
    // base-case report) emit the exact post-report coverage, not a
    // pre-report lower bound.
    if (cov_set_active)
      goto_coveraget::covered_run.store(
        tracked_instance, std::memory_order_relaxed);
    else
      goto_coveraget::live_reached.store(
        tracked_instance, std::memory_order_relaxed);

    // Item 2 / 2e final write-back: covered_set is the single live
    // accumulator (loaded input ∪ every edge the Item-2e hook persisted
    // as it was witnessed this run). Fold in any reached universe edge
    // defensively, then one final atomic rewrite. Monotone union —
    // never truncated; only true-P_SATISFIABLE edges are ever added.
    if (cov_set_active)
    {
      for (const auto &[cond, loc] : goto_coveraget::all_claims)
        if (reached_claims.count(cond + "\t" + loc))
          goto_coveraget::covered_set.emplace(cond, loc);
      goto_coveraget::write_covered_set_atomic();
      log_success(
        "coverage covered-set written to {}",
        goto_coveraget::covered_set_outpath);
    }
  }

  else if (is_branch_func_cov)
  {
    //! Might got incorrect total number when using --k-induction
    //! due to that the symex->goto_functions has been simplified
    const size_t total = goto_coveraget::total_func_branch;
    const size_t tracked_instance = count_reached_goals(reached_claims);
    log_success("\n[Coverage]\n");
    log_result("Function Entry Points & Branches : {}", total);
    log_result("Reached : {}", tracked_instance);

    // show claims
    if (options.get_bool_option("branch-function-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_claims)
        log_status("  {}", prettify_solidity_expr(claim));
    }

    if (total != 0)
      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    else
      log_result("Branch Coverage: N/A (no branches)");
  }

  else if (is_k_path_cov)
  {
    const size_t total = goto_coveraget::total_kpath;
    const size_t spanning = goto_coveraget::total_kpath_spanning;

    // Phase-2 (issue #4335): both numerator and denominator must restrict
    // to maximal goals under the atom-multiset subsumption order (Marré-
    // Bertolino, IEEE TSE 2003). Filter reached_claims against
    // k_path_spanning_redundant so a reached-but-subsumed goal does not
    // inflate the numerator against the maximal-only denominator.
    const size_t tracked_instance = std::count_if(
      reached_claims.begin(), reached_claims.end(), is_kpath_maximal);

    log_success("\n[Coverage]\n");
    log_result("k-Path Witnesses : {}", total);
    log_result("Spanning Set : {}", spanning);
    log_result("Reached : {}", tracked_instance);

    // Listing shows every reached claim regardless of maximality so the
    // user can see which subsumed goals were also reached — this is a
    // diagnostic flag, not a coverage-formula display.
    if (options.get_bool_option("k-path-coverage-claims"))
      for (const auto &claim : reached_claims)
        log_status("  {}", prettify_solidity_expr(claim));

    if (spanning != 0)
      log_result("k-Path Coverage: {}%", tracked_instance * 100.0 / spanning);
    else
      log_result("k-Path Coverage: N/A (no k-path goals)");
  }

  else if (is_path_cov)
  {
    // BEFORE any number is printed: does this run establish anything at all?
    //
    // Measured on a real benchmark, 120166 paths were instrumented and symex
    // produced ZERO verification conditions — the harness never entered a
    // single unit — and every path was then reported "U", which reads exactly
    // like an honest solver timeout. The audit aborts on that, and on the
    // per-unit version of it, so the coverage figures below are never printed
    // for a run that is vacuous rather than merely incomplete. It runs first
    // precisely so those figures do not appear and get quoted.
    //
    // This path is also reached with an empty reached-set when the equation
    // simplifies to zero remaining claims (see run_thread), which is exactly
    // the whole-run version of the same failure.
    goto_coveraget::audit_entry_liveness(options.get_option("focus-function"));

    // CERTIFICATION MODE PRINTS NO [Coverage] BLOCK AT ALL.
    //
    // The claims in this mode are `assume(box); assert(tr == pi)` — a certified
    // box makes them HOLD, so they would be counted as U/bounded-holds and the
    // run would read "Path Coverage: 0%" for a completely successful
    // certification. The verdict of a certification run is VERIFICATION
    // SUCCESSFUL versus FAILED, nothing else.
    //
    // Suppressed rather than annotated, on precedent: the path-distribution
    // line once mixed a per-round count into a structural measurement and
    // printed "1 path(s) total ... 0.33x" for a contract whose real figures are
    // 8 and 2.67x. It was documented in a comment and still got quoted, and a
    // measured result had to be retracted. A number that does not exist cannot
    // be quoted; a number with a disclaimer attached can.
    if (goto_coveraget::path_cov_outer_box_mode)
    {
      // Same reason as certification mode: these claims are ladder probes, and
      // a probe that HOLDS is a measurement, not an uncovered path. Reporting
      // them through the coverage counters would print a number that means
      // nothing and reads like one that does.
      goto_coveraget::report_outer_boxes();
    }
    else if (goto_coveraget::path_cov_certify_mode)
    {
      goto_coveraget::audit_certify_witness(
        options.get_bool_option("cov-report-json"));
      log_status(
        "--path-cov-certify: no [Coverage] block is printed in certification "
        "mode — the claims here are `assume(box); assert(tr == pi)`, so a "
        "CERTIFIED box makes them hold and would be counted as uncovered");
      // THE RESULT LINE, and it replaces the verdict line as this mode's
      // answer. The non-vacuity witness is REFUTED on every run that
      // certifies, so a certified box now prints VERIFICATION FAILED — which
      // is why the tool has to state its own result rather than let a caller
      // infer one. Printed last in this arm so it is the final word before the
      // verdict a reader must NOT use.
      goto_coveraget::report_path_cov_certify();
    }
    else if (goto_coveraget::path_cov_assert_mode)
    {
      // Same reason as the two arms above, and the sharpest case of it: in this
      // mode a claim that HOLDS is the WANTED outcome, so the coverage counters
      // would print "Path Coverage: 0%" for a completely successful ladder.
      // audit_entry_liveness above still runs and is the precondition that
      // stops a never-entered unit from making every candidate hold vacuously.
      goto_coveraget::report_path_cov_assertions();
    }
    else
    {
      // Denominator = the no-skip static universe built by
      // solidity_path_coverage() (one entry per enumerated complete path), so a
      // covered-set skip never shrinks it. Numerator = universe paths EITHER
      // witnessed this run OR already persisted by an earlier round.
      const size_t total = goto_coveraget::all_claims.size();
      // Complete-path coverage keys its cross-run cover on the
      // CONTENT-ADDRESSED stable id, not on `covered_set` (which
      // solidity_path_coverage() deliberately leaves empty — see
      // path_witnessed_earlier). Using covered_set here reported every path
      // carried over from an earlier round as U: a path with a counterexample
      // in hand, filed under "undecided".
      const bool cov_set_active = !goto_coveraget::path_covered_outpath.empty();
      size_t tracked_instance = 0;
      for (const auto &k : goto_coveraget::all_claims)
      {
        // THE SAME DISJUNCTION `Path Status: F` USES, and it has to be, because
        // `Reached` and `F` are two renderings of one set. Adding `v == 'F'`
        // below without adding it here produced a report reading
        // `Path Coverage: 0%` and `Path Status: F 1` in the same block --
        // measured on the injected mid-harvest fault. Two numbers computed from
        // one fact must not be able to disagree; a reader would have to guess
        // which was the defect.
        char vv = 0;
        {
          std::lock_guard lk(goto_coveraget::claim_outcome_mutex);
          auto it_v =
            goto_coveraget::claim_outcome.find(k.first + "\t" + k.second);
          if (it_v != goto_coveraget::claim_outcome.end())
            vv = it_v->second;
        }
        if (
          goto_coveraget::path_witnessed_earlier(k) ||
          reached_claims.count(k.first + "\t" + k.second) || vv == 'F')
          ++tracked_instance;
      }

      log_success("\n[Coverage]\n");
      // ---- THE COMPLETENESS LINE, printed in BOTH directions ----
      //
      // A partial report lands under the same filename a complete one does, and
      // the [Coverage] block below is byte-compatible with a run that genuinely
      // measured these numbers. So the discriminator is stated as its own line,
      // unconditionally: a marker that is only present when something is wrong
      // is indistinguishable, to any consumer that has not been taught about
      // it, from a marker that was forgotten.
      if (is_partial)
        log_result(
          "Report Completeness: PARTIAL — this run did not conclude ({}). {} "
          "of "
          "{} claim(s) had been decided. Every count below is a LOWER BOUND: "
          "the "
          "paths never reached are reported U with reason "
          "'run-died-before-solving', which is NOT the same as "
          "'not-solved-this-run'",
          partial_reason,
          goto_coveraget::live_decided.load(std::memory_order_relaxed),
          goto_coveraget::claims_total_atomic.load(std::memory_order_relaxed));
      else
        log_result("Report Completeness: COMPLETE");
      // Printed on EVERY path-coverage run, including when the budget is off
      // and when it never fired. "The cap was on" and "the cap fired N times"
      // are separate statements, and a run whose numbers were shaped by an
      // abandoned query must not look like one whose solver answered
      // everything.
      log_result(
        "Claim Budget: {} — {} claim(s) abandoned over budget ({})",
        goto_coveraget::claim_budget_seconds == 0
          ? std::string("unlimited")
          : std::to_string(goto_coveraget::claim_budget_seconds) +
              "s per claim",
        goto_coveraget::claim_budget_exceeded.load(std::memory_order_relaxed),
        goto_coveraget::claim_budget_mechanism.empty()
          ? std::string("no enforcement")
          : goto_coveraget::claim_budget_mechanism);
      // ---- WHAT THE ARITHMETIC RE-SOLVE COST, AND WHAT IT BOUGHT ----
      //
      // Printed on EVERY path-coverage run, including when the mechanism is off
      // and when it never fired. "The flag was on" and "it fired N times" are
      // separate statements, and a mechanism whose price is unmeasured is one
      // nobody can decide to keep -- nobody knew, when this was designed,
      // whether it would fire on three claims or three thousand.
      //
      // `conditions seen` is printed even at ZERO and especially then: zero
      // means no arithmetic check was enabled, or none reached these units, and
      // without it a run that re-solved nothing looks exactly like a run that
      // had nothing to re-solve.
      if (options.get_bool_option("path-cov-arith-resolve"))
        log_result(
          // `took the constrained witness` counts SAT re-solves, NOT wraps
          // fixed: a path whose original witness already satisfied every check
          // re-solves to an equally good one and is counted here too. Measured
          // on D10_WrapNotPanic, 3 of 3 -- and only ONE of those three was
          // wrapping. Naming it "replaced by a non-wrapping one" would have let
          // a reader infer three defects fixed from one.
          "Arithmetic Re-solve: {} condition(s) seen, {} claim(s) re-solved in "
          "{}s, {} took the constrained witness (this counts SAT re-solves, "
          "not "
          "wraps fixed -- a path whose witness was already fine is counted "
          "too), "
          "{} path(s) PROVEN reachable only through a checked-arithmetic "
          "revert, "
          "{} Foundry case(s) REFUSED for that reason",
          goto_coveraget::arith_conditions_seen.load(std::memory_order_relaxed),
          goto_coveraget::arith_resolve_queries.load(std::memory_order_relaxed),
          goto_coveraget::arith_resolve_ms.load(std::memory_order_relaxed) /
            1000.0,
          goto_coveraget::arith_resolve_replaced.load(
            std::memory_order_relaxed),
          goto_coveraget::arith_revert_only_paths.size(),
          goto_coveraget::arith_revert_only_suppressed.load(
            std::memory_order_relaxed));
      else
        log_result(
          "Arithmetic Re-solve: OFF (--path-cov-arith-resolve). A witnessed "
          "path whose counterexample wraps or divides by zero is emitted as a "
          "normal-exit test and is RED on the unmodified contract");

      // Printed unconditionally, zero included. A NON-ZERO value means the same
      // claim key reached the solve loop more than once and a decision it had
      // already made would have been thrown away -- so this line is both the
      // guard's effect and the duplicate-instrumentation defect's detector.
      log_result(
        "Verdicts Preserved: {} — a claim already DECIDED whose later solve "
        "returned no verdict kept its decision. Non-zero also means the same "
        "claim key was solved more than once, which is a separate defect",
        goto_coveraget::verdicts_preserved.load(std::memory_order_relaxed));

      // ---- THE OTHER DIRECTION, which the line above cannot see ----
      //
      // `Verdicts Preserved` fires only when the LATER solve returned no
      // verdict. A later solve that returns a DIFFERENT verdict passes it
      // silently, and that is the common case: measured on a 20-line contract
      // at
      // --solidity-max-tx 2, one key was solved twice and got PASSED then
      // FAILED while `Verdicts Preserved` read 0.
      //
      // Printed unconditionally, zeros included, and as three separate numbers
      // because they answer three different questions: how much solving was
      // repeated, how often a decision was superseded, and what the worst key
      // was. A single "duplicates: N" would let "no duplication" and "the
      // counter was never wired" look the same.
      log_result(
        "Claim Multiplicity: {} extra solve(s) beyond the first across all "
        "keys; worst key '{}' decided {} time(s); ceiling {}; {} decided "
        "verdict(s) superseded by a different decided verdict. Repetition is "
        "EXPECTED and is not bounded by this run's tx/unwind: one "
        "instantiation "
        "per transaction, one per re-entry level, and one per caller for a "
        "public unit another unit calls internally. The path's answer is the "
        "disjunction over them. The number to read is SUPERSEDED: non-zero "
        "means "
        "one key's instantiations disagreed, so what this run reports for that "
        "path depended on which was solved first",
        path_cov_extra_solves.load(std::memory_order_relaxed),
        path_cov_max_solves_key.empty()
          ? std::string("(none)")
          : prettify_solidity_expr(path_cov_max_solves_key),
        path_cov_max_solves,
        path_cov_allowed_solves == 0
          ? std::string("not enforced (--path-cov-max-claim-solves unset)")
          : std::to_string(path_cov_allowed_solves),
        path_cov_verdict_upgrades.load(std::memory_order_relaxed));

      if (total == 0)
        log_result("No complete path enumerated");
      else
      {
        log_result("Complete Paths : {}", total);
        log_result("Reached : {}", tracked_instance);
        log_result("Path Coverage: {}%", tracked_instance * 100.0 / total);
        // Exit-shape breakdown on stdout as well as in the JSON. The JSON is a
        // file, and the regression harness matches program OUTPUT only —
        // without this line the whole exit classification (and in particular
        // the refusal to call an undetermined exit "normal") cannot be
        // regression-locked.
        size_t n_rev = 0, n_und = 0;
        for (const auto &k : goto_coveraget::all_claims)
        {
          if (
            goto_coveraget::revert_paths.count(k) ||
            goto_coveraget::rollback_revert_paths.count(k))
            ++n_rev;
          else if (goto_coveraget::undetermined_exit_paths.count(k))
            ++n_und;
        }
        log_result(
          "Path Exits: normal {}, revert {}, undetermined {}",
          total - n_rev - n_und,
          n_rev,
          n_und);

        // Tri-state totals on stdout too, for the same reason as the exit line:
        // the per-path F/I/U verdicts live only in the JSON file, which the
        // regression harness cannot inspect. Without this the central honesty
        // rule — a claim that merely HELD within the bound is U, never I — is
        // not regression-locked. Same rule as the JSON emission below.
        {
          // See path_cov_can_prove_unreachable(): no coverage configuration can
          // establish unreachability, so this is false and every non-refuted
          // path is U. Deliberately NOT keyed on --solidity-max-tx 0, which the
          // back-edge-to-SKIP pass reduces to a single guarded transaction.
          const bool unb = path_cov_can_prove_unreachable();
          size_t nF = 0, nI = 0, nU = 0;
          // Every U carries a reason token. See path_u_reason_token(): a U cell
          // that can silently absorb an implementation defect has already done
          // so three times, and each time the broken case looked exactly like
          // an honest timeout.
          std::map<std::string, size_t> u_reasons;
          for (const auto &t : goto_coveraget::path_u_reason_tokens())
            u_reasons[t] = 0;
          std::vector<std::string> untokened;
          for (const auto &k : goto_coveraget::all_claims)
          {
            const std::string sig = k.first + "\t" + k.second;
            char v = 0;
            {
              std::lock_guard lk(goto_coveraget::claim_outcome_mutex);
              auto it_o = goto_coveraget::claim_outcome.find(sig);
              if (it_o != goto_coveraget::claim_outcome.end())
                v = it_o->second;
            }
            // ---- A VERDICT THAT WAS MADE MUST NOT BE LOST ----
            //
            // `v == 'F'` is in this disjunction because the two records of a
            // refutation are written at DIFFERENT TIMES and a dying run can
            // land between them. `claim_outcome[sig] = 'F'` is written the
            // instant the solver answers SAT; `reached_claims.emplace(sig)` is
            // written only after build_goto_trace, the counterexample harvest
            // and every artifact emitter have run. Everything in that window
            // can throw.
            //
            // MEASURED on notes/coverage/poc/P16_Mapping.sol -- a 30-line
            // nested mapping, 8 paths, `std::bad_alloc` at 4 GB:
            //
            //     ✗ FAILED: 'put:path:7 at'
            //     Path Status: F 0, I 0, U 8
            //     ERROR: INTERNAL DEFECT — 1 path(s) are reported U with NO
            //            reason token: sol:@C@P16_Mapping@F@put#31:path:7
            //
            // The run printed the refutation and then reported zero
            // refutations. The path fell to the `else` below,
            // `path_u_reason_token` has no token for 'F' (correctly -- an F is
            // not a U), and the invariant fired. So the abort was RIGHT and its
            // cause was here: a witnessed path filed as undecided, which on a
            // live run reads exactly like an honest "we could not decide it".
            //
            // The payload may be absent for such a claim -- the harvest is what
            // died -- and that is reported as it is: `status: F` with an empty
            // `inputs` is the true statement (a witness exists, its values did
            // not survive), whereas dropping the F asserts something false
            // about the program. The same disjunction is applied to `witnessed`
            // in the JSON block so the file and the terminal cannot disagree.
            if (
              goto_coveraget::path_witnessed_earlier(k) ||
              reached_claims.count(sig) || v == 'F')
              ++nF;
            else if (v == 'P' && unb)
              ++nI;
            else
            {
              ++nU;
              const std::string tok = goto_coveraget::path_u_reason_token(k);
              if (tok.empty())
                untokened.push_back(k.first);
              else
                ++u_reasons[tok];
            }
          }
          log_result("Path Status: F {}, I {}, U {}", nF, nI, nU);
          // Printed whenever there is a U, with EVERY slot listed including the
          // zeros: a category that stops occurring is noticed, a category that
          // silently disappears from the output is not.
          if (nU > 0)
          {
            std::string breakdown;
            for (const auto &t : goto_coveraget::path_u_reason_tokens())
              breakdown += (breakdown.empty() ? "" : ", ") + t + " " +
                           std::to_string(u_reasons[t]);
            log_result("U Reasons: {}", breakdown);
          }
          if (!untokened.empty())
          {
            std::string names;
            for (const auto &n : untokened)
              names += (names.empty() ? "" : "; ") + n;
            log_error(
              "--solidity-path-coverage: INTERNAL DEFECT — {} path(s) are "
              "reported U with NO reason token: {}. The claim this pass makes "
              "is "
              "that every uncovered path carries a named reason and there is "
              "no "
              "unexplained remainder. An untokened U is that claim being "
              "false, "
              "and it is exactly the shape in which an implementation defect "
              "hides inside an honest-looking 'we do not know'.",
              untokened.size(),
              names);
            abort();
          }
        }
      }

      // Final write-back of the CONTENT-ADDRESSED cover.
      //
      // This call was missing, and the omission killed the whole cross-run
      // mechanism silently. `write_path_covered_set_atomic` existed and was
      // correct; nothing called it, while the code here called the BRANCH
      // metric's writer, gated on `covered_set_outpath` — which
      // solidity_path_coverage() never sets. Measured before the fix: running
      // with `--coverage-covered-set cov.json` produced no file at all, so
      // every round re-instrumented everything and the escalation story (round
      // N spends its budget only on paths still lacking a CE) did not work at
      // all.
      //
      // Same defect shape as the entry-liveness audit before it was wired: a
      // function that is written, tested by eye, and never called. The function
      // itself selects only 'F' claims, so nothing but a real witness is ever
      // persisted.
      if (cov_set_active)
      {
        goto_coveraget::write_path_covered_set_atomic("at run end");
        log_success(
          "coverage covered-set written to {}",
          goto_coveraget::path_covered_outpath);
      }
    } // end of the non-certification reporting block
  }

  // Generate JSON coverage report
  if (options.get_bool_option("cov-report-json"))
  {
    using json = nlohmann::json;

    std::string cov_type = "unknown";
    if (is_branch_cov)
      cov_type = "branch";
    else if (is_branch_func_cov)
      cov_type = "branch-function";
    else if (is_k_path_cov)
      cov_type = "k-path";
    else if (is_path_cov)
      cov_type = "solidity-path";
    else if (is_cond_cov)
      cov_type = "condition";
    else if (is_assert_cov)
      cov_type = "assertion";

    const auto &all_claims = goto_coveraget::all_claims;
    std::set<std::string> source_files;
    json claims_json = json::array();

    // Bound under which THIS run's verdicts were produced. Recorded on every
    // path entry: a "holds" verdict is only meaningful together with the
    // exploration it was obtained under.
    const std::string max_tx = options.get_option("solidity-max-tx");
    const std::string unwind_s = options.get_option("unwind");
    bool loops_truncated = false;
    {
      std::lock_guard lk(goto_functionst::truncated_loops_mutex);
      loops_truncated = !goto_functionst::truncated_loops.empty();
    }
    // See path_cov_can_prove_unreachable(). This used to be
    // `max_tx == "0" && !loops_truncated`, on the belief that max_tx 0 gives an
    // unbounded transaction sequence. It does not: coverage rewrites the
    // dispatcher back-edge to a SKIP, so max_tx 0 explores ONE transaction —
    // fewer than max_tx 2. Every non-refuted path is therefore U.
    const bool unbounded_run = path_cov_can_prove_unreachable();
    // What the transaction driver ACTUALLY explored, stated on every entry so a
    // reader never has to infer it from the flag. `--solidity-max-tx 0` is the
    // trap: it reads as "unbounded" and is in fact the shallowest setting.
    const std::string tx_exploration =
      max_tx == "0"
        ? std::string(
            "one guarded transaction: --solidity-max-tx 0 emits the "
            "`while (nondet) dispatch()` driver, but Solidity coverage "
            "rewrites "
            "every _ESBMC_Main* back-edge to a SKIP, so the loop runs at most "
            "once. This explores FEWER transactions than --solidity-max-tx 2")
        : (max_tx.empty()
             ? std::string(
                 "coverage default: the dispatcher loop back-edge is rewritten "
                 "to a SKIP, leaving one guarded transaction")
             : ("exactly " + max_tx +
                " straight-line transaction(s) (no dispatcher back-edge to "
                "remove)"));

    for (const auto &[claim_msg, claim_loc] : all_claims)
    {
      std::string claim_sig = claim_msg + "\t" + claim_loc;
      bool covered = reached_claims.count(claim_sig) > 0;

      // For assertion coverage, check reached_mul_claims instead
      if (is_assert_cov)
        covered = reached_mul_claims.count(claim_sig) > 0;

      json loc = parse_claim_location(claim_loc);
      std::string file = loc["file"];
      if (!file.empty())
        source_files.insert(file);

      json claim_entry;
      claim_entry["condition"] = prettify_solidity_expr(claim_msg);
      claim_entry["file"] = loc["file"];
      claim_entry["line"] = loc["line"];
      claim_entry["column"] = loc["column"];
      claim_entry["function"] = loc["function"];
      claim_entry["status"] = covered ? "covered" : "uncovered";
      // k-path Phase-2 (#4335): annotate each claim as feasible (a
      // maximal element of the subsumption lattice and thus part of the
      // spanning set) or spanning-set-redundant (subsumed by a stronger
      // emitted goal — covering it adds no information beyond covering
      // its subsumer).
      if (is_k_path_cov)
      {
        const auto &redundant = goto_coveraget::k_path_spanning_redundant;
        claim_entry["feasibility"] = redundant.count({claim_msg, claim_loc}) > 0
                                       ? "spanning-set-redundant"
                                       : "feasible";
      }

      // Solidity complete-path coverage: the CE->generalisation interface.
      // Tri-state per complete path, plus the bound the verdict was reached
      // under and how the path exits.
      if (is_path_cov)
      {
        char v = 0;
        {
          std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
          auto it_o = goto_coveraget::claim_outcome.find(claim_sig);
          if (it_o != goto_coveraget::claim_outcome.end())
            v = it_o->second;
        }
        // Witnessed = refuted this run, or already refuted in an earlier
        // escalation round (cross-run covered-set). Either way a concrete
        // input exists for this path.
        const bool prior =
          goto_coveraget::path_witnessed_earlier({claim_msg, claim_loc});
        // `v == 'F'`: the same disjunction the stdout counters use, for the
        // same reason. The refutation is recorded in `claim_outcome` the moment
        // the solver answers and in `reached_claims` only after the trace and
        // harvest have run, so a run that dies in between has a verdict that
        // `covered` cannot see. Keeping the two readers in step is what stops
        // the file and the terminal from disagreeing about how many paths were
        // witnessed. See the stdout block for the measured P16_Mapping case.
        const bool witnessed = covered || prior || v == 'F';

        // F: feasible, CE in hand.
        // I: PROVEN unreachable. No coverage configuration can establish this
        //    today — see path_cov_can_prove_unreachable() — so this arm is
        //    currently unreachable by construction, deliberately.
        // U: everything else, INCLUDING a claim that "holds" at this
        //    exploration: that means no witness within one-or-N transactions
        //    from the post-constructor entry state, not that the path is
        //    unreachable. It stays U and is flagged `bounded_holds`.
        std::string tri;
        if (witnessed)
          tri = "F";
        else if (v == 'P' && unbounded_run)
          tri = "I";
        else
          tri = "U";
        claim_entry["status"] = tri;
        if (tri == "U")
        {
          // Same single source as the stdout breakdown, so the file and the
          // terminal can never disagree about why a path is uncovered.
          const std::string tok =
            goto_coveraget::path_u_reason_token({claim_msg, claim_loc});
          claim_entry["u_reason"] = tok;
          if (tok == "named-obstacle")
          {
            auto ob =
              goto_coveraget::named_obstacle_paths.find({claim_msg, claim_loc});
            if (ob != goto_coveraget::named_obstacle_paths.end())
              claim_entry["u_reason_detail"] = ob->second;
          }
          else if (tok == "unit-not-entered")
          {
            // WHY the unit was not entered, not just that it wasn't: excluded
            // on purpose by --focus-function is a normal per-method run, while
            // "the harness never entered it" is a defect. A consumer that
            // cannot tell them apart would treat a deliberate narrowing as a
            // broken run, or worse, the reverse.
            const size_t p = claim_msg.rfind(":path:");
            const std::string unit =
              p == std::string::npos ? claim_msg : claim_msg.substr(0, p);
            auto ue = goto_coveraget::units_not_entered.find(unit);
            if (ue != goto_coveraget::units_not_entered.end())
              claim_entry["u_reason_detail"] = ue->second;
          }
        }
        // Kept alongside the token: a consumer that already reads these keys
        // keeps working, and `bounded_holds` additionally marks the U's worth
        // re-checking under a deeper exploration.
        if (v == 'P' && !unbounded_run)
          claim_entry["bounded_holds"] = true;
        if (!witnessed && v == 0)
          // Never handed to the solver this run (sliced away, or skipped
          // because an earlier round already covered a different path).
          claim_entry["not_solved_this_run"] = true;

        claim_entry["bound"]["max_tx"] = max_tx.empty() ? "default" : max_tx;
        claim_entry["bound"]["unwind"] =
          unwind_s.empty() ? "default" : unwind_s;
        claim_entry["bound"]["kind"] = unbounded_run ? "unbounded" : "bounded";
        claim_entry["bound"]["tx_exploration"] = tx_exploration;
        if (loops_truncated)
          claim_entry["bound"]["loops_truncated"] = true;
        // Both revert shapes exit the transaction, so both are "revert": the
        // custom-error one (ASSUME(false) in a #sol_error callee) and the
        // rollback one (require/revert("msg"), which restores `*this` and then
        // reaches END_FUNCTION). Reporting the latter as "normal" would claim a
        // reverting transaction succeeded.
        const bool ck_err =
          goto_coveraget::revert_paths.count({claim_msg, claim_loc}) > 0;
        const bool ck_rb = goto_coveraget::rollback_revert_paths.count(
                             {claim_msg, claim_loc}) > 0;
        const bool ck_un = goto_coveraget::undetermined_exit_paths.count(
                             {claim_msg, claim_loc}) > 0;
        claim_entry["exit_kind"] =
          (ck_err || ck_rb) ? "revert" : (ck_un ? "undetermined" : "normal");
        if (ck_rb)
          // Distinguishes the two: here the rollback IS modelled, so
          // final_state is the correctly restored post-state.
          claim_entry["revert_kind"] = "rollback";
        else if (ck_err)
          claim_entry["revert_kind"] = "custom-error";
        else if (ck_un)
          claim_entry["exit_kind_undetermined_reason"] =
            "reaches END_FUNCTION bypassing the function epilogue and carries "
            "no rollback restore: a `require` failing before any state write "
            "and a plain early `return` compile to the same shape, so this is "
            "either a revert or a normal early exit";
        claim_entry["witnessed_in_earlier_round"] = prior && !covered;

        // PROVEN reachable only through a checked-arithmetic revert. Emitted
        // beside `status: F` rather than instead of it: the path IS feasible
        // and a witness exists, it is just a witness the chain reaches through
        // a Panic. A consumer that renders this path as a bare call asserting a
        // normal exit produces a RED test, which is the single outcome this
        // pipeline must never produce -- so the flag is published rather than
        // left to be inferred from the values.
        if (
          goto_coveraget::arith_revert_only_paths.count({claim_msg, claim_loc}))
        {
          claim_entry["arith_revert_only"] = true;
          claim_entry["arith_revert_only_reason"] =
            "the re-solve of this claim under the enabled arithmetic check "
            "conditions is UNSAT, which PROVES that no input reaches this path "
            "without violating a checked operation. On chain the path is "
            "therefore reached through a Panic revert (0x11 overflow / 0x12 "
            "division by zero), NOT through the normal exit this entry's "
            "exit_kind reports -- exit_kind classifies the MODEL, and the "
            "model has no Panic. Render with vm.expectRevert, or do not render "
            "it at all; a bare call asserting a normal exit is red on the "
            "unmodified contract";
        }

        // path_id == enc(pi): the integer encoding the path's whole decision
        // sequence (tr accumulator). Text before ":path:" is the function.
        const auto pos = claim_msg.rfind(":path:");
        if (pos != std::string::npos)
        {
          claim_entry["path_id"] = claim_msg.substr(pos + 6);
          claim_entry["path_function"] = claim_msg.substr(0, pos);
        }
        // The decision depth, i.e. the `cnt` half of the path's identity. A
        // stage-2 query is `tr == enc && cnt == depth`, so a consumer that only
        // had `path_id` could not build one; it had to be told the depth out of
        // band, which is the opposite of an interface.
        {
          auto dp =
            goto_coveraget::path_decision_depth.find({claim_msg, claim_loc});
          if (dp != goto_coveraget::path_decision_depth.end())
            claim_entry["path_depth"] = dp->second;

          // ---- THE ORDERED DECISION SEQUENCE, for F claims only ----
          //
          // This is what makes path coverage comparable with branch coverage at
          // all: the two metrics share no denominator until the witnessed paths
          // are projected onto the DECISIONS they walk. `path_id` cannot be
          // projected — `enc` records the arms and nothing about the sites, so
          // only the enumerator knows which instruction each bit came from.
          //
          // RESTRICTED TO F on purpose, and it is not merely a saving. F is the
          // only status the projection can use (an uncovered path witnesses no
          // decision), and F is rare — single digits per unit — while a unit
          // can enumerate 120166 paths. Emitting for every claim would put the
          // report's size in the same order as the path count for no gain.
          if (tri == "F" && pos != std::string::npos)
          {
            const std::string unit = claim_msg.substr(0, pos);
            const uint64_t penc =
              strtoull(claim_msg.substr(pos + 6).c_str(), nullptr, 10);
            auto ti = goto_coveraget::path_decision_table.find(unit);
            auto xi = goto_coveraget::path_decision_index.find(unit);
            if (
              ti != goto_coveraget::path_decision_table.end() &&
              xi != goto_coveraget::path_decision_index.end() &&
              dp != goto_coveraget::path_decision_depth.end() && dp->second > 0)
            {
              const uint64_t pdepth = dp->second;
              json seq = json::array();
              size_t missing = 0;
              for (uint64_t k = 0; k < pdepth; ++k)
              {
                const uint64_t key = penc >> (pdepth - 1 - k);
                auto di = xi->second.find(key);
                if (di == xi->second.end() || di->second >= ti->second.size())
                {
                  // Reported as a hole, never skipped. A shorter array would
                  // read as "this path walks fewer decisions" — a claim about
                  // the path, when the truth is a claim about the recording.
                  json h;
                  h["index"] = k + 1;
                  h["unrecorded_prefix_enc"] = std::to_string(key);
                  seq.push_back(h);
                  ++missing;
                  continue;
                }
                const auto &d = ti->second[di->second];
                const bool arm_taken = (key & 1) != 0;
                json e;
                e["index"] = k + 1;
                json dloc = parse_claim_location(d.loc);
                e["file"] = dloc["file"];
                e["line"] = dloc["line"];
                e["column"] = dloc["column"];
                e["function"] = dloc["function"];
                e["operand"] = d.sub;
                e["arm"] = arm_taken ? "taken" : "fall-through";
                // This arm's claim text, already inverted (assert(P) fails when
                // P is false, so assert(guard) covers the FALL-THROUGH edge).
                // Published rather than left to the consumer: inverting it
                // there is a silent error that still produces a number.
                //
                // DIAGNOSTIC ONLY — do not join on it. Measured: a `require`
                // lowers to a guard one `not` deeper under path coverage than
                // under branch coverage (the revert-observation gate), so the
                // texts differ for the same decision while a plain `if` agrees
                // verbatim. See path_decisiont in goto_coverage.h. Join on
                // file+line, which is the comparison metric's own unit.
                e["branch_claim"] = prettify_solidity_expr(
                  arm_taken ? d.cond_arm_true : d.cond_arm_false);
                if (d.synthetic_abi_gate)
                  // No branch-coverage counterpart AND a location copied from
                  // the unit's first body instruction, so a consumer matching
                  // on location alone would credit a real decision on that
                  // line.
                  e["synthetic_abi_gate"] = true;
                seq.push_back(e);
              }
              claim_entry["decisions"] = seq;
              if (missing > 0)
                claim_entry["decisions_unrecorded"] = missing;
            }

            // ---- R0's EVENT RUNG: the emits this path walks, in order ----
            //
            // Read out of the published tables with exactly the same prefix
            // walk as the decisions above, so the two fields cannot disagree
            // about which path they describe. The inner map is keyed by program
            // position, and std::map iterates it in ascending key order, which
            // IS program order — that is the whole reason the recorder keys on
            // position rather than appending.
            //
            // WRITTEN EVEN WHEN EMPTY, and that is the point of the field. R0
            // is "same exit, same revert reason, SAME EVENTS IN THE SAME
            // ORDER"; a generator can only assert the third rung if it can tell
            // "this path emits nothing" apart from "nobody looked". An absent
            // key here means recording was off; `[]` means it was on and the
            // path emitted nothing recordable.
            //
            // ⚠ "nothing recordable" is not "nothing emitted". The qualified
            // spelling `emit L.E(x)` is dropped in the front end and never
            // reaches the goto program, so a consumer must not turn `[]` into
            // an assertion that NO event fires. It may only assert the events
            // that ARE listed, in the order listed.
            {
              auto ei = goto_coveraget::path_event_table.find(unit);
              auto vi = goto_coveraget::path_event_index.find(unit);
              if (
                ei != goto_coveraget::path_event_table.end() &&
                vi != goto_coveraget::path_event_index.end() &&
                dp != goto_coveraget::path_decision_depth.end())
              {
                json evs = json::array();
                for (uint64_t k = dp->second + 1; k-- > 0;)
                {
                  auto pit = vi->second.find(penc >> k);
                  if (pit == vi->second.end())
                    continue;
                  for (const auto &[evpos, evid] : pit->second)
                  {
                    (void)evpos;
                    if (evid < ei->second.size())
                      evs.push_back(ei->second[evid]);
                  }
                }
                claim_entry["events"] = evs;
              }
            }
          }
        }

        // CE payload: the concrete values behind an F. Absent for I/U (there
        // is no counterexample to report).
        std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
        auto it_ce = goto_coveraget::path_ce.find(claim_sig);
        if (it_ce != goto_coveraget::path_ce.end())
        {
          const auto &ce = it_ce->second;
          json ins = json::object();
          for (const auto &[n, v] : ce.inputs)
            ins[prettify_solidity_expr(n)] = v;
          json envj = json::object();
          for (const auto &[n, v] : ce.env)
            envj[prettify_solidity_expr(n)] = v;
          json fin = json::object();
          for (const auto &[n, v] : ce.final_state)
            fin[prettify_solidity_expr(n)] = v;
          claim_entry["inputs"] = ins;
          claim_entry["env"] = envj;
          // What the outside world returned on this path, in call order. Kept
          // separate from `inputs` because a consumer can CHOOSE an input and
          // cannot choose this: a replay has to mock the callee to return these
          // values. An ARRAY, not a map — repeated calls reuse the same symbol
          // name and their order is part of the answer.
          json ext = json::array();
          for (const auto &[n, v] : ce.extcall_returns)
            ext.push_back(
              {{"symbol", prettify_solidity_expr(n)}, {"value", v}});
          claim_entry["extcall_returns"] = ext;
          if (ce.extcall_returns.empty())
            // NOT "there were no external calls": say what is actually known.
            //
            // This list is ALWAYS empty: nothing anywhere writes to
            // ce.extcall_returns. Below is what stands between the value and
            // this field, measured on a four-unit contract whose units differ
            // ONLY in the syntactic shape the call's result arrives in. There
            // are THREE distinct reasons, not one, and a fix for any single one
            // leaves the field empty in the other shapes -- which looks exactly
            // like no fix at all.
            //
            //   (a) result bound to a named local (`bool ok = c.f();`) and the
            //       assembly form (`success` assigned inside an approximated
            //       block): the step DOES reach the harvest and
            //       get_nondet_symbol DOES resolve it. It is then dropped
            //       because the classification below has exactly three
            //       outcomes -- parameter -> inputs, environment -> env,
            //       otherwise -> dropped_internal -- and a call's return is a
            //       LOCAL, so it lands in the third. There is no bucket for it.
            //   (b) low-level `(bool ok, ) = a.call("")`: get_nondet_symbol
            //       returns nil, so the step is skipped BEFORE classification.
            //       (This is the mechanism an earlier version of this comment
            //       named -- correctly, but for this shape only.)
            //   (c) used inline (`if (c.f())`): no named local is ever
            //       assigned, so no step carries the value at all.
            //
            // The _ESBMC_Nondet_Extcall_* symbols that ARE in the trace are the
            // re-entry model's method-choice bits, not the returned value
            // (measured: identical on two paths that disagree about it), so
            // they are deliberately not reported here.
            claim_entry["ce_extraction"]["extcall_returns_unavailable_reason"] =
              "not implemented yet, and for three different reasons depending "
              "on the syntactic shape the call's result arrives in; this field "
              "does NOT say which one applies to this claim. (a) bound to a "
              "named local, or assigned inside an approximated assembly block: "
              "the value IS harvested and resolved, then dropped because the "
              "classification has buckets only for parameters and environment "
              "values and a call's return is a local. (b) low-level "
              "`(bool ok, ) = a.call(...)`: get_nondet_symbol returns nil, so "
              "the step is skipped before classification. (c) used inline, "
              "`if (c.f())`: no named local is assigned, so no trace step "
              "carries it. The _ESBMC_Nondet_Extcall_* symbols that ARE in the "
              "trace are the re-entry model's method-choice bits, not the "
              "returned value (measured: identical on two paths that disagree "
              "about it), so they are deliberately not reported here. An empty "
              "list means UNKNOWN, not 'this path performs no external call'";
          // Contract state this path STARTED from. Without it a path guarded by
          // state an earlier transaction established cannot be replayed from
          // the inputs alone.
          json entry = json::object();
          for (const auto &[n, v] : ce.entry_storage)
            entry[prettify_solidity_expr(n)] = v;
          claim_entry["entry_storage"] = entry;
          if (!ce.entry_storage_known)
            // Never let an empty entry_storage be read as "the contract started
            // empty": say that the marker was not observed instead.
            claim_entry["ce_extraction"]["entry_storage_unavailable_reason"] =
              "the entry marker for this path's function was not seen in the "
              "trace, so the state it started from was not captured; an empty "
              "entry_storage here means UNKNOWN, not 'no state'";
          // A custom-error revert in a scope the revert-observation gate does
          // NOT cover (constructor / library / free function) is still modelled
          // by ASSUME(false) with no rollback, so the harvested values are the
          // state AT the revert, not the post-state — on-chain the transaction
          // is undone and the post-state is the pre-call state. Publishing
          // those under `final_state` would be a wrong value carrying a
          // disclaimer; publish them under their true meaning instead and leave
          // `final_state` empty, since the post-state is genuinely unavailable.
          if (ce.revert_pre_rollback)
          {
            claim_entry["state_at_revert_point"] = fin;
            claim_entry["final_state"] = json::object();
          }
          else
            claim_entry["final_state"] = fin;
          // ---- THE UNIT'S OWN RETURN VALUE, BESIDE THE POST-STATE ----
          //
          // Never inside `final_state`: a return value is not state, and a
          // consumer that asserted it as state would assert the wrong thing
          // about the wrong object. This is the field the PUT emitter needs to
          // turn `c0.f(args);` into `assertEq(c0.f(args), v);`.
          //
          // `return_value_known` is emitted in BOTH directions. Absent, or an
          // empty string on its own, would read as "this unit returns nothing"
          // -- a claim about the CONTRACT manufactured out of an absence in the
          // harvest. The three ways to be unknown are genuinely different and
          // the reason names all of them.
          claim_entry["return_value_known"] = ce.return_value_known;
          if (ce.return_value_known)
            claim_entry["return_value"] = ce.return_value;
          else
            claim_entry["ce_extraction"]["return_value_unavailable_reason"] =
              "no scalar return value was captured on this path. THREE "
              "different situations produce this and the field does not say "
              "which: (a) the unit returns nothing; (b) it returns an "
              "aggregate, tuple or dynamic type, which the instrumenter does "
              "not materialise because there is no single renderable value; "
              "(c) this path exits without reaching a RETURN at all, which is "
              "the normal shape of a revert. An absent value means UNKNOWN, "
              "never 'this unit returns nothing'";
          // State this path WROTE but whose value is not renderable (mapping /
          // dynamic-array stores). Listed so "absent from final_state" is never
          // read as "unchanged".
          if (!ce.state_written_unrendered.empty())
          {
            json un = json::array();
            for (const auto &n : ce.state_written_unrendered)
              un.push_back(prettify_solidity_expr(n));
            claim_entry["state_written_value_unavailable"] = un;
          }
          // Not silent: say how many nondet values were classified as harness
          // plumbing and left out of `inputs`/`env`.
          claim_entry["ce_extraction"]["harness_nondets_dropped"] =
            ce.dropped_internal;
          // How the values were harvested. REQUIRED for the reader to
          // interpret an empty final_state: with `sliced` true the symex
          // slicer kept only what the path claim depends on — and a path
          // claim's guard mentions only the ghost accumulators — so state
          // writes are legitimately absent rather than non-existent. Re-run
          // with --no-slice to obtain the post-state.
          claim_entry["ce_extraction"]["sliced"] = ce.sliced;
          claim_entry["ce_extraction"]["payload_symbols_exempt_from_slicing"] =
            ce.payload_symbols_protected;
          claim_entry["ce_extraction"]["compact_trace"] = ce.compact_trace;
          claim_entry["ce_extraction"]["scoped_to_claim"] = ce.scoped_to_claim;
          if (!ce.scoped_to_claim)
            claim_entry["ce_extraction"]["post_state_may_include_later_tx"] =
              "this path's own assert was not found in the trace, so the whole "
              "trace was scanned; with more than one transaction the reported "
              "post-state may belong to a later transaction";
          // A custom-error revert is modelled by ASSUME(false) in the error
          // callee with NO state rollback, so the harvested post-state is the
          // state at the revert point. On-chain the transaction reverts and
          // every write is undone, i.e. the true post-state is the PRE-state.
          // Say so rather than let a consumer read these values as the result.
          if (ce.revert_pre_rollback)
            claim_entry["ce_extraction"]["post_state_unavailable_reason"] =
              "custom-error revert in a scope without rollback modelling "
              "(constructor / library / free function): the harvested values "
              "are published as `state_at_revert_point`, NOT as final_state. "
              "The real post-state of a reverted transaction is the pre-call "
              "state, which this model does not reconstruct";
          // Only meaningful when the payload symbols were NOT exempted: with
          // the exemption in place an empty final_state means this path writes
          // no state, and saying "the writes were sliced away" would be a
          // wrong explanation of a correct result.
          if (ce.sliced && !ce.payload_symbols_protected && fin.empty())
            claim_entry["ce_extraction"]["final_state_unavailable_reason"] =
              "slicing active and the payload symbols were not exempted: a "
              "path claim depends only on the ghost path accumulators, so "
              "contract state writes are sliced out of the counterexample. "
              "Ask for --cov-report-json (which exempts them) or re-run with "
              "--no-slice";

          // ---- EVERY WITNESS, WHEN THERE IS MORE THAN ONE ----
          //
          // `witness_count` is emitted unconditionally, because "this path has
          // one input tuple that reaches it" and "this path has sixteen and we
          // reported one" are different statements and the report could not
          // previously make either. The `witnesses` ARRAY is emitted only above
          // 1: at count 1 it would duplicate the fields directly above it on
          // every F claim of every run, and the report is already ~1.6 MB on a
          // 2846-path contract.
          //
          // The extra witnesses are the raw material the stage-2 ladder wants
          // -- sibling spans need more than one point in a path's domain, and
          // one counterexample cannot bracket a boundary.
          auto it_all = goto_coveraget::path_ce_all.find(claim_sig);
          const size_t n_wit = it_all == goto_coveraget::path_ce_all.end()
                                 ? 1
                                 : it_all->second.size();
          claim_entry["ce_extraction"]["witness_count"] = n_wit;
          if (n_wit > 1)
          {
            json wits = json::array();
            for (const auto &w : it_all->second)
            {
              json wj;
              json wi = json::object();
              for (const auto &[n, v] : w.inputs)
                wi[prettify_solidity_expr(n)] = v;
              json we = json::object();
              for (const auto &[n, v] : w.env)
                we[prettify_solidity_expr(n)] = v;
              json wf = json::object();
              for (const auto &[n, v] : w.final_state)
                wf[prettify_solidity_expr(n)] = v;
              json wen = json::object();
              for (const auto &[n, v] : w.entry_storage)
                wen[prettify_solidity_expr(n)] = v;
              wj["inputs"] = wi;
              wj["env"] = we;
              wj["entry_storage"] = wen;
              // Same rule as the single-witness block above: a revert without
              // rollback modelling has no post-state, and publishing the
              // values at the revert point under `final_state` would be a
              // wrong value carrying a disclaimer.
              if (w.revert_pre_rollback)
                wj["state_at_revert_point"] = wf;
              else
                wj["final_state"] = wf;
              wits.push_back(wj);
            }
            claim_entry["witnesses"] = wits;
          }
        }
        else if (tri == "F")
        {
          // Witnessed, but not by THIS run: the cross-run covered-set already
          // held it, so the path was not re-instrumented and no model was
          // produced here.
          //
          // The covered set now carries the PAYLOAD alongside the id, so this
          // is a lookup rather than an excuse. The old text said the values
          // "are in the report of the round that witnessed it" — which was a
          // hope, not a fact: that report is a file in a directory this run
          // knows nothing about, and on a run that died before writing one it
          // never existed. Emit the persisted values when they are there, and
          // keep an accurate reason when they are not.
          const auto *prior_ce =
            goto_coveraget::path_payload_earlier({claim_msg, claim_loc});
          if (prior_ce != nullptr)
          {
            json ins = json::object();
            for (const auto &[n, v] : prior_ce->inputs)
              ins[prettify_solidity_expr(n)] = v;
            json envj = json::object();
            for (const auto &[n, v] : prior_ce->env)
              envj[prettify_solidity_expr(n)] = v;
            json fin = json::object();
            for (const auto &[n, v] : prior_ce->final_state)
              fin[prettify_solidity_expr(n)] = v;
            json entry = json::object();
            for (const auto &[n, v] : prior_ce->entry_storage)
              entry[prettify_solidity_expr(n)] = v;
            claim_entry["inputs"] = ins;
            claim_entry["env"] = envj;
            claim_entry["entry_storage"] = entry;
            if (prior_ce->revert_pre_rollback)
            {
              claim_entry["state_at_revert_point"] = fin;
              claim_entry["final_state"] = json::object();
            }
            else
              claim_entry["final_state"] = fin;
            claim_entry["ce_extraction"]["sliced"] = prior_ce->sliced;
            claim_entry["ce_extraction"]["compact_trace"] =
              prior_ce->compact_trace;
            claim_entry["ce_extraction"]["scoped_to_claim"] =
              prior_ce->scoped_to_claim;
            claim_entry["ce_extraction"]["harness_nondets_dropped"] =
              prior_ce->dropped_internal;
            // NAMED, not silent. These values were harvested under a DIFFERENT
            // run's bound and slicing configuration, which is recorded in the
            // payload itself; a consumer that treats them as this run's own
            // output would attribute this run's `bound` block to them.
            claim_entry["ce_extraction"]["payload_source"] =
              "persisted by the earlier round that witnessed this path "
              "(cross-run covered-set), not harvested by this run; the "
              "bound/slicing flags in this block are that round's";
          }
          else
            claim_entry["ce_extraction"]["payload_absent_reason"] =
              "path was already witnessed in an earlier round (covered-set) "
              "and therefore not re-instrumented this run, and the covered-set "
              "file carries NO payload for it. This report cannot produce a "
              "test for this path; re-run without the covered-set, or with a "
              "covered-set written by a build that persists payloads";
        }
      }
      claims_json.push_back(claim_entry);
    }

    size_t total = all_claims.size();
    size_t covered_count = 0;
    for (const auto &c : claims_json)
      // Path coverage reports the tri-state in `status`; "F" is its
      // "covered" (a complete path with a counterexample in hand).
      if (c["status"] == "covered" || c["status"] == "F")
        covered_count++;

    // For k-path coverage, restrict the summary to maximal goals so the
    // JSON percentage matches the terminal spanning-set-filtered output.
    // Individual claims keep their `feasibility` annotation so consumers
    // that want raw counts can still derive them from the `claims` array.
    if (is_k_path_cov)
    {
      total = 0;
      covered_count = 0;
      for (const auto &c : claims_json)
        if (c["feasibility"] == "feasible")
        {
          ++total;
          if (c["status"] == "covered")
            ++covered_count;
        }
    }

    json report;
    // ---- PARTIAL, AT THE TOP LEVEL, IN BOTH DIRECTIONS ----
    //
    // A partial report is written to the same `cov-report.json` a complete one
    // is, because that is where every consumer looks and the alternative --
    // writing it somewhere else -- reproduces today's behaviour of keeping
    // nothing. The marker is therefore the ONLY thing separating them, so it is
    // emitted unconditionally: `false` on a complete run is what makes its
    // absence a detectable defect rather than an assumed default. A consumer
    // that reads `partial` with a default of false would otherwise treat a
    // report from a build that forgot to set it as complete.
    //
    // Duplicated under `summary` on purpose. Every existing reader of these
    // reports (report_summary.py, branch_gate.py, fset_cmp.py, gap_attribution
    // .py) opens `summary` and several never look at the top level at all; a
    // marker they cannot see is a marker that does not exist.
    report["partial"] = is_partial;
    if (is_partial)
      report["partial_reason"] = partial_reason;
    report["coverage_type"] = cov_type;
    report["source_files"] = json::array();
    for (const auto &f : source_files)
      report["source_files"].push_back(f);
    report["claims"] = claims_json;
    report["summary"]["partial"] = is_partial;
    if (is_partial)
    {
      report["summary"]["partial_reason"] = partial_reason;
      // The two numbers that say HOW partial. Without them a reader can see
      // that the run was cut short but not whether it was cut short at 1% or
      // 99%, and the difference decides whether the report is worth consuming
      // at all. Taken from the atomics the job loop maintains, so they are the
      // same numbers the signal handler would have printed.
      report["summary"]["claims_decided"] =
        goto_coveraget::live_decided.load(std::memory_order_relaxed);
      report["summary"]["claims_total"] =
        goto_coveraget::claims_total_atomic.load(std::memory_order_relaxed);
    }
    report["summary"]["total"] = total;
    report["summary"]["covered"] = covered_count;
    report["summary"]["uncovered"] = total - covered_count;
    report["summary"]["percentage"] =
      total > 0 ? covered_count * 100.0 / total : 0.0;
    // For branch / branch-function modes: explicit "no_branch" marker when the
    // source has zero branches. Key absent when total > 0 — consumers should
    // read with default false. Not emitted for condition/assertion/k-path
    // modes (their zero-claim case has different semantics).
    if (total == 0 && (is_branch_cov || is_branch_func_cov))
      report["summary"]["no_branch"] = true;

    // Path coverage: tri-state totals + the run's bound, so a consumer can
    // see at a glance how many complete paths have a CE (F), how many are
    // PROVEN unreachable (I, unbounded runs only), and how many are still
    // open (U) — with `bounded_holds` counting the U's that held within this
    // bound and are therefore worth re-checking unbounded.
    if (is_path_cov)
    {
      size_t nF = 0, nI = 0, nU = 0, nBH = 0, nRevert = 0;
      for (const auto &c : claims_json)
      {
        const std::string s = c["status"];
        if (s == "F")
          ++nF;
        else if (s == "I")
          ++nI;
        else
          ++nU;
        if (c.contains("bounded_holds"))
          ++nBH;
        if (c.value("exit_kind", "") == "revert")
          ++nRevert;
      }
      report["summary"]["paths_total"] = total;
      report["summary"]["F_feasible_with_ce"] = nF;
      report["summary"]["I_proven_unreachable"] = nI;
      report["summary"]["U_undecided"] = nU;
      report["summary"]["U_of_which_bounded_holds"] = nBH;

      // ---- HOW MANY WITNESSES THE REPORT ACTUALLY CARRIES ----
      //
      // Counted from `claims_json`, i.e. from what was EMITTED, not from the
      // producer's own map: this pass has already shipped one recorder that ran
      // on every path and was consumed by nothing, and a census taken from the
      // producer would have looked healthy throughout. Printed on stdout too,
      // because the regression harness matches program output only, so a field
      // that exists solely inside the file cannot be regression-locked.
      {
        size_t wtotal = 0, wmulti = 0;
        for (const auto &c : claims_json)
        {
          if (c.value("status", "") != "F")
            continue;
          const size_t n =
            c.contains("ce_extraction")
              ? c["ce_extraction"].value("witness_count", (size_t)1)
              : (size_t)1;
          wtotal += n;
          if (n > 1)
            ++wmulti;
        }
        report["summary"]["witnesses_total"] = wtotal;
        report["summary"]["F_with_multiple_witnesses"] = wmulti;
        log_status(
          "--solidity-path-coverage: CE PAYLOADS published for {} witness(es) "
          "across {} witnessed path(s); {} path(s) carry more than one "
          "(--all-witnesses is {}). Every witness of a path is a further point "
          "in that path's input domain, which is what a sibling span needs and "
          "what one counterexample cannot give",
          wtotal,
          nF,
          wmulti,
          options.get_bool_option("all-witnesses") ? "on" : "off");
      }

      // ---- THE DECISION-SEQUENCE PUBLICATION, COUNTED ON STDOUT ----
      //
      // A mechanism whose only evidence is a field inside a file nobody reads
      // has no evidence at all: this pass has already shipped one recorder that
      // ran on every path and was consumed by nothing. Counting it here, from
      // `claims_json` (i.e. from what was ACTUALLY emitted, not from what the
      // producer believes it emitted), makes "it fired" and "it fired on N
      // things" the same statement.
      //
      // `without` is printed even when it is zero, and it is the number that
      // matters: an F carrying no sequence cannot be projected onto decisions,
      // so it is a silent hole in any comparison built on this field.
      {
        size_t with = 0, without = 0, steps = 0, holes = 0, synth = 0;
        for (const auto &c : claims_json)
        {
          if (c.value("status", "") != "F")
            continue;
          if (!c.contains("decisions"))
          {
            ++without;
            continue;
          }
          ++with;
          for (const auto &d : c["decisions"])
          {
            ++steps;
            if (d.contains("unrecorded_prefix_enc"))
              ++holes;
            if (d.value("synthetic_abi_gate", false))
              ++synth;
          }
        }
        report["summary"]["decision_sequences"]["paths_with"] = with;
        report["summary"]["decision_sequences"]["paths_without"] = without;
        report["summary"]["decision_sequences"]["decision_steps"] = steps;
        report["summary"]["decision_sequences"]["unrecorded_steps"] = holes;
        report["summary"]["decision_sequences"]["synthetic_abi_gate_steps"] =
          synth;
        log_status(
          "--solidity-path-coverage: DECISION SEQUENCES published for {} of {} "
          "witnessed path(s) ({} step(s), {} unrecorded, {} synthesised ABI "
          "value gate). A path with no sequence cannot be projected onto "
          "decisions, and the synthesised gate has no branch-coverage "
          "counterpart, so both are named rather than folded in",
          with,
          nF,
          steps,
          holes,
          synth);
      }

      // ---- THE EVENT SEQUENCES, COUNTED THE SAME WAY AND FOR THE SAME REASON
      // ----
      //
      // Counted from `claims_json`, i.e. from what was emitted, so "the rung
      // exists" and "the rung fired on N paths" are one statement.
      //
      // THE THREE NUMBERS ARE NOT REDUNDANT, and the middle one is the whole
      // point. `with_field` counts F claims carrying the array AT ALL;
      // `with_events` counts those whose array is NON-EMPTY. A contract that
      // emits nothing and a recorder that never fired both give
      // `with_events = 0` — they are told apart only by `without_field`, which
      // is 0 in the first case and nF in the second. Folding the two into one
      // count is exactly how an always-empty channel passes for a working one.
      {
        size_t with_field = 0, with_events = 0, without_field = 0, evsteps = 0;
        for (const auto &c : claims_json)
        {
          if (c.value("status", "") != "F")
            continue;
          if (!c.contains("events"))
          {
            ++without_field;
            continue;
          }
          ++with_field;
          const size_t n = c["events"].size();
          evsteps += n;
          if (n > 0)
            ++with_events;
        }
        report["summary"]["event_sequences"]["paths_with_field"] = with_field;
        report["summary"]["event_sequences"]["paths_without_field"] =
          without_field;
        report["summary"]["event_sequences"]["paths_with_events"] = with_events;
        report["summary"]["event_sequences"]["event_steps"] = evsteps;
        log_status(
          "--solidity-path-coverage: EVENT SEQUENCES published for {} of {} "
          "witnessed path(s), {} of them non-empty ({} emit(s) total). An "
          "EMPTY array is a measurement (\"this path emits nothing "
          "recordable\") and a MISSING field is not, so the two are counted "
          "apart; and `recordable` excludes the qualified spelling `emit "
          "L.E(x)`, which the front end drops",
          with_field,
          nF,
          with_events,
          evsteps);
      }
      // Every U's reason, with all slots present including the zeros — the
      // summary must never let a category vanish by simply not appearing.
      {
        json ur = json::object();
        for (const auto &t : goto_coveraget::path_u_reason_tokens())
          ur[t] = 0;
        for (const auto &c : claims_json)
          if (c.value("status", "") == "U")
          {
            const std::string tok = c.value("u_reason", "");
            if (!tok.empty())
              ur[tok] = ur[tok].get<size_t>() + 1;
          }
        report["summary"]["U_reasons"] = ur;
      }
      report["summary"]["revert_exit_paths"] = nRevert;
      // Its own cell beside F/I/U, never folded into any of them: "this path
      // needs an overflow" is a DECIDED property, and the cost of deciding it
      // travels with it so a reader can tell a run that measured this from a
      // run that did not ask.
      report["summary"]["arith_resolve"]["enabled"] =
        options.get_bool_option("path-cov-arith-resolve");
      report["summary"]["arith_resolve"]["conditions_seen"] =
        goto_coveraget::arith_conditions_seen.load(std::memory_order_relaxed);
      report["summary"]["arith_resolve"]["claims_resolved"] =
        goto_coveraget::arith_resolve_queries.load(std::memory_order_relaxed);
      report["summary"]["arith_resolve"]["seconds"] =
        goto_coveraget::arith_resolve_ms.load(std::memory_order_relaxed) /
        1000.0;
      // SAT re-solves, not wraps fixed. See the stdout line for why the
      // distinction is worth a key name this long.
      report["summary"]["arith_resolve"]["took_constrained_witness"] =
        goto_coveraget::arith_resolve_replaced.load(std::memory_order_relaxed);
      report["summary"]["arith_revert_only_paths"] =
        goto_coveraget::arith_revert_only_paths.size();
      report["summary"]["bound"]["max_tx"] =
        max_tx.empty() ? "default" : max_tx;
      report["summary"]["bound"]["unwind"] =
        unwind_s.empty() ? "default" : unwind_s;
      report["summary"]["bound"]["kind"] =
        unbounded_run ? "unbounded" : "bounded";
      report["summary"]["bound"]["tx_exploration"] = tx_exploration;
      // THE PER-CLAIM BUDGET IS PART OF THE BOUND, not a footnote. A capped
      // run's U counts are not comparable with an uncapped run's: some of its
      // U's mean "we stopped asking", and a reader diffing two reports without
      // this field would read that as "no witness exists". Emitted in both
      // directions (0 = unlimited) for the same reason `partial` is.
      report["summary"]["bound"]["claim_timeout_s"] =
        goto_coveraget::claim_budget_seconds;
      report["summary"]["bound"]["claim_timeout_enforcement"] =
        goto_coveraget::claim_budget_mechanism.empty()
          ? std::string("unlimited: no per-claim budget was applied")
          : goto_coveraget::claim_budget_mechanism;
      report["summary"]["claims_abandoned_over_budget"] =
        goto_coveraget::claim_budget_exceeded.load(std::memory_order_relaxed);
      if (!unbounded_run)
        report["summary"]["note"] =
          "no coverage configuration can establish unreachability, so I is "
          "never emitted and every path that merely held at this exploration "
          "is reported as U with bounded_holds=true. In particular "
          "--solidity-max-tx 0 is NOT an unbounded run: it emits the "
          "`while (nondet) dispatch()` driver, whose back-edge Solidity "
          "coverage then rewrites to a SKIP, leaving one guarded transaction "
          "— strictly fewer than --solidity-max-tx 2. Use a larger "
          "--solidity-max-tx N to explore deeper, not 0";
      // Known modelling limitation, stated rather than left for the reader to
      // rediscover: the entry state of a transaction is whatever the
      // constructor left behind. State variables are NOT havoc'd, so a path
      // guarded by state that only a PRIOR transaction can establish (a
      // balance, a role, an initialised flag) is unreachable at this
      // transaction bound and lands in U — not because the path is dead, but
      // because the run never built the state that unlocks it. Raising
      // --solidity-max-tx explores more of those; it does not remove the
      // limitation.
      report["summary"]["known_limitation_entry_state"] =
        "transaction entry state is the post-constructor state; contract state "
        "is not havoc'd, so paths guarded by state that an earlier transaction "
        "would have to establish are reported U at this tx bound. A U is "
        "therefore not evidence that the path is unreachable";
    }

    std::ofstream out("cov-report.json");
    out << report.dump(2) << std::endl;
    log_success("Coverage report written to cov-report.json");

    // Seal the journal. Everything before this point wrote `complete: false`,
    // which is the honest state of a file that is refreshed while the solve is
    // still running. A consumer must be able to tell the journal of a run that
    // finished from the journal of a run that was killed, and the only moment
    // the tool can say so is here — beside the report it did manage to write.
    if (is_path_cov && !goto_coveraget::path_ce_journal_path.empty())
      goto_coveraget::write_path_ce_journal_atomic(
        "at run end", /*complete=*/true);
  }

  cov_block_reported = true;

  // Generate pytest test case from collected data (for coverage mode)
  if (options.get_bool_option("generate-pytest-testcase"))
  {
    std::string input_file = options.get_option("input-file");
    std::string module_name = pytest_generator::extract_module_name(input_file);
    std::string pytest_filename =
      pytest_generator::generate_pytest_filename(module_name);
    pytest_gen.set_values_only(options.get_bool_option("pytest-values-only"));
    pytest_gen.generate(pytest_output_dir(options), pytest_filename);
  }

  // Generate CTest test cases from collected data (for coverage mode)
  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate(ctest_output_dir(options));
  }

  // Generate Foundry test cases from collected data (for coverage mode)
  if (options.get_bool_option("generate-foundry-testcase"))
    foundry_gen.generate();
}

/* Closing line of a coverage run, in place of a verification verdict: it says
 * whether the percentages above were actually measured. Without it a run that
 * solved none of its goals — the solver erred, --multi-fail-fast cut the run
 * short, --smt-formula-only never solved anything — still prints a percentage
 * that reads as measured (issue #6387). Both outcomes exit 0: an incomplete
 * measurement is not a program defect. */
void report_coverage_completeness()
{
  // Nothing was measured, so there is nothing to qualify.
  if (!cov_block_reported)
    return;

  std::lock_guard lock(cov_report_mutex);

  // A coverage run reports no violations. Anything it did refute would be
  // lost without this, so name it: the user asked for a measurement, not for
  // silence about a bug ESBMC happened to find on the way.
  const auto &suppressed = cov_suppressed_violations;
  if (!suppressed.empty())
  {
    log_warning(
      "\n{} claim(s) outside the coverage instrumentation were violated. A "
      "coverage run does not verify the program, so these are not reported as "
      "failures; re-run without the coverage flag to see them:",
      suppressed.size());
    for (const auto &claim : suppressed)
      log_warning("  {}", claim);
  }

  const auto &reasons = cov_incomplete_reasons;
  if (reasons.empty())
  {
    log_success("\nCOVERAGE ANALYSIS COMPLETE");
    return;
  }

  const size_t undecided = undecided_cov_goals;
  if (undecided > 0)
    log_fail(
      "\nCOVERAGE ANALYSIS INCOMPLETE: {} goal(s) undecided; the percentages "
      "above are lower bounds",
      undecided);
  else
    log_fail(
      "\nCOVERAGE ANALYSIS INCOMPLETE: the percentages above are lower "
      "bounds");
  for (const auto &reason : reasons)
    log_fail("  reason: {}", reason);
}

// Output coverage information whenever an instrumented assertion is found
// violated. It is helpful when the program is too large and ESBMC cannot
// finish, we can still get some info about the coverage
void bmct::report_coverage_verbose(
  const claim_slicer &claim,
  const std::string &claim_sig,
  const bool &is_assert_cov,
  const bool &is_cond_cov,
  const bool &is_branch_cov,
  const bool &is_branch_func_cov,
  const bool &is_k_path_cov,
  const std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims)
{
  // for condition coverage verbose output
  // total_cond: the combination of assertion's guard and location, which is
  // used to identify each assertion in multi-property checking.

  auto current_pair = std::make_pair(claim.claim_msg, claim.claim_loc);

  if (is_cond_cov)
  {
    auto total_cond = goto_coveraget::total_cond;

    if (total_cond.count(current_pair))
    {
      if (
        options.get_bool_option("condition-coverage-claims") ||
        options.get_bool_option("condition-coverage-claims-rm"))
      {
        // show claims
        log_status("\n  {} : SATISFIED", prettify_solidity_expr(claim_sig));
      }

      // show coverage data
      log_result(
        "Current Condition Coverage: {}%\n",
        reached_claims.size() * 100.0 / total_cond.size());
    }
  }
  else
  {
    if (is_assert_cov)
    {
      const size_t total_instance = goto_coveraget::total_assert_ins;
      const size_t tracked_instance = reached_mul_claims.size();

      if (options.get_bool_option("assertion-coverage-claims"))
      {
        for (const auto &claim : reached_mul_claims)
          log_status("  {}", prettify_solidity_expr(claim));
      }
      if (total_instance != 0)
      {
        if (total_instance >= tracked_instance)
          log_result(
            "Assertion Instances Coverage: {}%",
            tracked_instance * 100.0 / total_instance);
        else
          log_result("Assertion Instances Coverage: 0%");
      }
    }
    else if (is_branch_cov)
    {
      size_t totals = goto_coveraget::total_branch;
      const int tracked_instance = reached_claims.size();
      // show claims
      if (options.get_bool_option("branch-coverage-claims"))
      {
        // reached claims:
        for (const auto &claim : reached_claims)
          log_status("  {}", prettify_solidity_expr(claim));
      }

      if (totals != 0)
        log_result("Branch Coverage: {}%", tracked_instance * 100.0 / totals);
      else
        log_result("Branch Coverage: 0%");
    }
    else if (is_branch_func_cov)
    {
      size_t totals = goto_coveraget::total_func_branch;
      const int tracked_instance = reached_claims.size();
      // show claims
      if (options.get_bool_option("branch-function-coverage-claims"))
      {
        // reached claims:
        for (const auto &claim : reached_claims)
          log_status("  {}", prettify_solidity_expr(claim));
      }

      if (totals != 0)
        log_result(
          "Branch Function Coverage: {}%", tracked_instance * 100.0 / totals);
      else
        log_result("Branch Function Coverage: 0%");
    }
    else if (is_k_path_cov)
    {
      // Match the final reporter's spanning-set formula so per-witness
      // progress agrees with the final summary.
      const size_t tracked_instance = std::count_if(
        reached_claims.begin(), reached_claims.end(), is_kpath_maximal);

      if (options.get_bool_option("k-path-coverage-claims"))
        log_status("\n  {} : SATISFIED", prettify_solidity_expr(claim_sig));

      // spanning >= 1 here: verbose only fires after a reached claim,
      // which implies total_kpath >= 1 (Marré-Bertolino).
      log_result(
        "Current k-Path Coverage: {}%\n",
        tracked_instance * 100.0 / goto_coveraget::total_kpath_spanning);
    }
    else
    {
      log_error("Unsupported coverage metrics");
      abort();
    }
  }
}

bool bmct::proves_the_program() const
{
  return !vacuity_detected && !ltl_uninstrumented && run_fully_proved(options);
}

void bmct::report_result(smt_resultt &res)
{
  // k-induction prints its own messages
  if (options.get_bool_option("k-induction-parallel"))
    return;
  // Diagnostic pass: report_property_verdicts already prints the per-property
  // results; suppress any global verdict from this level.
  if (options.get_bool_option("diagnose-unknown-properties"))
    return;
  // A coverage run replaced the program's assertions with reachability
  // probes, so it neither proved nor refuted anything about the program.
  // Its result is the [Coverage] block, not a verification verdict.
  if (
    options.get_bool_option("coverage-measurement") &&
    !goto_coveraget::path_cov_query_mode())
    return;

  // Dead-code analysis is advisory. Its instrumented reachability probes are
  // violated (SAT) for every *live* branch, which would otherwise drive the
  // verdict to FAILED. The CWE-561 findings are reported separately by
  // report_dead_code(); a completed analysis is a successful run, so never
  // flip the verdict (SV-COMP compatibility, issue #4495). A solver error
  // still surfaces so we don't claim success over an incomplete analysis.
  if (options.get_bool_option("dead-code-check"))
  {
    if (res == P_SMTLIB)
      return; // only a formula/VCC was emitted; no verdict to report
    if (res == P_ERROR)
    {
      log_error("SMT solver failed");
      return;
    }
    report_success();
    return;
  }

  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");
  bool mul = reports_multi_property_verdict(options);

  switch (res)
  {
  case P_UNSATISFIABLE:
    if (is && term)
    {
      report_failure();
    }
    else if (!bs || mul)
    {
      // Suppress spurious success when a violation was already found in a
      // previous k step (multi-property sequential k-induction).  The final
      // verdict is printed by do_bmc_strategy once the loop terminates.
      //
      // Also suppress when symex flipped `disable-inductive-step` mid-run
      // (recursion, threads, function-pointer calls): the IS encoding is
      // incomplete, so its UNSAT does not prove safety. is_inductive_step
      // _violated checks the same flag and returns UNKNOWN, so reporting
      // SUCCESSFUL here would contradict the strategy-level verdict.
      if (
        !options.get_bool_option("kind-violation-found") &&
        !(is && options.get_bool_option("disable-inductive-step")))
      {
        // A bounded round proves nothing on its own: the driver decides the
        // verdict once the search becomes exhaustive.
        if (options.get_bool_option("suppress-bounded-success"))
          log_status("No violation found within the current context bound");
        else if (!proves_the_program())
          report_unknown();
        else
          report_success();
      }
    }
    else
    {
      log_status("No bug has been found in the base case");
    }
    break;

  case P_SATISFIABLE:
    if (!is && !fc)
    {
      report_violation();
    }
    else if (fc)
    {
      log_status("The forward condition is unable to prove the property");
    }
    else if (is)
    {
      log_status("The inductive step is unable to prove the property");
    }
    break;

    // SMTLIB-only emission: nothing was actually checked, so return without
    // reporting any verdict.
  case P_SMTLIB:
    return;

  default:
    log_error("SMT solver failed");
    break;
  }

  if ((interleaving_number > 0) && options.get_bool_option("all-runs"))
  {
    log_status("Number of generated interleavings: {}", interleaving_number);
    log_status("Number of failed interleavings: {}", interleaving_failed);
  }
}

smt_resultt bmct::start_bmc()
{
  std::shared_ptr<symex_target_equationt> eq;
  smt_resultt res = run(eq);

  // The dead-code advisory is emitted here, once, rather than from
  // report_coverage inside multi_property_check: that runs per thread
  // interleaving, and reporting there called a branch dead on the strength of
  // the first interleaving alone. goto_functionst::reached_claims is a static
  // that is never cleared between interleavings, so by this point it holds
  // every probe reached by any of them. Emitting before report_result keeps the
  // [Dead code] section above the verdict, and routing the dead-store
  // advisories through the same call keeps both sets in one SARIF document
  // (issue #4495). Only a run that actually solved the probes can say anything
  // about dead code. --show-vcc returns P_SMTLIB from run_thread before
  // multi_property_check ever runs, and a solver failure gives P_ERROR; either
  // way reached_claims is empty while all_claims is full, so every branch would
  // be reported dead. report_result already declines to claim success over
  // those two results — stay silent here for the same reason.
  if (
    options.get_bool_option("dead-code-check") && res != P_SMTLIB &&
    res != P_ERROR)
  {
    report_dead_code(
      options, goto_functionst::reached_claims, dead_store_advisories);
    dead_store_sarif_written = true;
  }

  // multi-property traces are output during the run(eq); the verdicts are
  // held back until every interleaving has been explored
  if (!options.get_bool_option("multi-property"))
    report_trace(res, *eq);

  // A phase that died before its per-claim loop, or only emitted a formula,
  // skipped every property and never sets report_incomplete.
  if (phase_left_properties_undecided(report_incomplete, res))
    goto_functionst::property_verdicts.note_incomplete();
  else if (is_bounded_round(options))
    goto_functionst::property_verdicts.complete_round();

  // A single monolithic UNSAT refutes the disjunction of every claim's
  // violation, so on a genuinely conclusive run each claim holds. Anything
  // weaker leaves the properties this phase never separated out as NotChecked.
  if (all_properties_proved(res))
    goto_functionst::property_verdicts.promote_unchecked_to_passed();

  // The report describes the whole run, so an iterative strategy prints it
  // with its final verdict rather than once per k. --multi-property has
  // always reported per phase, and keeps doing so where a phase decided
  // something -- except under a k-step strategy, which accumulates into one
  // table that the concluding phase, or the driver, prints once.
  if (reports_final_verdict(res) || reports_multi_property_verdict(options))
    report_property_verdicts(res);
  report_result(res);

  // Dead-store advisories are verdict-independent, but the trace paths that
  // emit them (successful_trace / error_trace) do not run on every verdict —
  // e.g. a FAILED run under --no-cex / --result-only, or an SMTLIB-only
  // emission. Emit an advisory-only SARIF document here if none was written
  // with a trace, so the advisory is not silently dropped (the textual
  // advisory prints unconditionally in the driver).
  if (
    !dead_store_sarif_written && !dead_store_advisories.empty() &&
    !options.get_option("sarif-output").empty())
  {
    goto_tracet empty_trace;
    sarif_goto_trace(options, ns, empty_trace, dead_store_advisories);
    dead_store_sarif_written = true;
  }

  if (symex)
  {
    cs_bound_pruned = symex->cs_bound_pruned;
    symex->report_reduction_stats();
  }

  // The run's result doubles as the process exit code, and smt_resultt has no
  // unknown value. Every other path that prints UNKNOWN -- vacuity here, the
  // k-induction strategy elsewhere -- exits 0, because an unknown verdict does
  // not witness a bug. Map it here rather than rewrite the solver's own answer
  // where the verdict is decided (issue #7480).
  return verdict_is_unknown ? P_UNSATISFIABLE : res;
}

size_t bmct::barren_interleaving_budget() const
{
  const std::string budget = options.get_option("multi-property-interleavings");
  if (budget.empty())
    return default_barren_interleaving_budget;

  const long value = strtol(budget.c_str(), nullptr, 10);
  if (value < 1)
  {
    log_error("the value of multi-property-interleavings should be positive!");
    abort();
  }

  return value;
}

/// A k-step strategy clears the store once, before its first phase, and
/// starts a round at each base case.
static void prepare_property_verdicts(const optionst &options)
{
  if (reports_multi_property_verdict(options))
    goto_functionst::property_verdicts.clear();
  else if (is_bounded_round(options))
    goto_functionst::property_verdicts.begin_round();
}

smt_resultt bmct::run(std::shared_ptr<symex_target_equationt> &eq)
{
  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  const bool multi_property = options.get_bool_option("multi-property");
  prepare_property_verdicts(options);
  report_incomplete = false;

  if (options.get_bool_option("schedule"))
    return run_thread(eq);

  // Under --multi-property a violation no longer ends the run: a property
  // after the violated one may only be reachable in a later interleaving, and
  // stopping here leaves it unreported (discussion #6391). Keep exploring
  // until this many consecutive interleavings reach a verdict on nothing the
  // run had not already reached one on.
  const size_t barren_budget =
    multi_property ? barren_interleaving_budget() : 0;
  size_t barren_interleavings = 0;
  size_t verdicts_seen = 0;
  bool violation_seen = false;

  smt_resultt res;
  do
  {
    if (++interleaving_number > 1)
      log_status("Thread interleavings {}", interleaving_number);

    // Clear the cache between thread interleavings to prevent
    // incorrect caching of assertions with different thread contexts
    if (!options.get_bool_option("no-cache-asserts"))
      get_ssa_caching_db().clear();

    fine_timet bmc_start = current_time();
    res = run_thread(eq);

    if (res == P_SATISFIABLE)
    {
      if (config.options.get_bool_option("smt-model"))
        runtime_solver->print_model();

      if (config.options.get_bool_option("bidirectional"))
        bidirectional_search(*runtime_solver, *eq);
    }

    if (res)
    {
      if (res == P_SATISFIABLE)
        ++interleaving_failed;

      // --dead-code-check has to see every interleaving before it can call a
      // branch dead: each *live* probe comes back SAT, so stopping here would
      // leave every branch that is only reachable under a later thread ordering
      // looking unreached, and report it as CWE-561. There is no
      // early-exit-on-bug to preserve for this mode — the verdict is forced
      // SUCCESSFUL regardless (issue #4495). A solver error or an SMT-formula
      // emission still stops immediately: those are not "live probe" results
      // and must propagate. It also leaves violation_seen clear, so the barren
      // budget below never cuts the search short: --dead-code-check turns
      // --multi-property on implicitly, and it wants every interleaving.
      const bool keep_exploring_for_dead_code =
        options.get_bool_option("dead-code-check") && res == P_SATISFIABLE;

      if (!options.get_bool_option("all-runs") && !keep_exploring_for_dead_code)
      {
        // An error or an SMTLIB-only emission says nothing about the
        // remaining interleavings; only a violation is worth continuing past.
        // A violation already found stands: an undecided later interleaving
        // does not retract it. P_SMTLIB is excluded deliberately -- an
        // SMT-LIB-only emission must never be turned into a verdict.
        if (!multi_property || res != P_SATISFIABLE)
        {
          const bool keep = violation_seen && res == P_ERROR;
          report_incomplete = keep;
          return keep ? P_SATISFIABLE : res;
        }

        violation_seen = true;
      }
    }
    fine_timet bmc_stop = current_time();

    log_status("BMC program time: {}s", time2string(bmc_stop - bmc_start));

    // Only run for one run
    if (options.get_bool_option("interactive-ileaves"))
      return res;

    if (violation_seen)
    {
      const size_t verdicts_now = goto_functionst::property_verdicts.size();
      barren_interleavings =
        verdicts_now > verdicts_seen ? 0 : barren_interleavings + 1;
      verdicts_seen = verdicts_now;

      if (barren_interleavings >= barren_budget)
      {
        report_incomplete = true;
        break;
      }
    }

  } while (symex->setup_next_formula());

  if (options.get_bool_option("ltl"))
  {
    // So, what was the lowest value ltl outcome that we saw? The lattice runs
    // ⊥ < ⊥ᵖ < ⊤ᵖ < ⊤, and the two lower values say the property is violated
    // on some prefix, so they have to reach the process result rather than
    // only a log line.
    if (ltl_results_seen[ltl_res_bad])
    {
      log_result("Final lowest outcome: LTL_BAD");
      res = P_SATISFIABLE;
    }
    else if (ltl_results_seen[ltl_res_failing])
    {
      log_result("Final lowest outcome: LTL_FAILING");
      res = P_SATISFIABLE;
    }
    else if (ltl_results_seen[ltl_res_succeeding])
    {
      log_result("Final lowest outcome: LTL_SUCCEEDING");
      res = P_UNSATISFIABLE;
    }
    else if (ltl_results_seen[ltl_res_good])
    {
      log_result("Final lowest outcome: LTL_GOOD");
      res = P_UNSATISFIABLE;
    }
    else
    {
      // No outcome at all: either nothing was instrumented, or symex never
      // reached the monitor. Either way the property was not checked, which
      // report_result turns into UNKNOWN.
      log_warning("No LTL outcome seen; the property was not checked");
      ltl_uninstrumented = true;
      res = P_UNSATISFIABLE;
    }
  }

  return interleaving_failed > 0 ? P_SATISFIABLE : res;
}

void bmct::bidirectional_search(
  smt_convt &smt_conv,
  const symex_target_equationt &eq)
{
  // We should only analyze the inductive step's cex and we're running
  // in k-induction mode
  if (!(options.get_bool_option("inductive-step") &&
        options.get_bool_option("k-induction")))
    return;

  // We'll walk list of SSA steps and look for inductive assignments
  std::vector<stack_framet> frames;
  unsigned assert_loop_number = 0;
  for (const auto &ssait : eq.SSA_steps)
  {
    if (
      ssait.is_assert() && !is_nil_expr(ssait.cond_expr) &&
      smt_conv.l_get(ssait.cond_expr).is_false())
    {
      if (!ssait.loop_number)
        return;

      // Save the location of the failed assertion
      frames = ssait.stack_trace();
      assert_loop_number = ssait.loop_number;

      // We are not interested in instructions before the failed assertion yet
      break;
    }
  }

  for (auto f : frames)
  {
    // Look for the function
    goto_functionst::function_mapt::iterator fit =
      symex->goto_functions.function_map.find(f.function);
    assert(fit != symex->goto_functions.function_map.end());

    // Find function loops
    goto_loopst loops(f.function, symex->goto_functions, fit->second);

    if (!loops.get_loops().size())
      continue;

    auto lit = loops.get_loops().begin(), lie = loops.get_loops().end();
    while (lit != lie)
    {
      auto loop_head = lit->get_original_loop_head();

      // Skip constraints from other loops
      if (loop_head->loop_number == assert_loop_number)
        break;

      ++lit;
    }

    if (lit == lie)
      continue;

    // Get the loop vars
    auto all_loop_vars = lit->get_modified_loop_vars();
    all_loop_vars.insert(
      lit->get_unmodified_loop_vars().begin(),
      lit->get_unmodified_loop_vars().end());

    // Now, walk the SSA and get the last value of each variable before the loop
    std::unordered_map<irep_idt, std::pair<expr2tc, expr2tc>, irep_id_hash>
      var_ssa_list;

    for (const auto &ssait : eq.SSA_steps)
    {
      if (ssait.loop_number == lit->get_original_loop_head()->loop_number)
        break;

      if (ssait.ignore)
        continue;

      if (!ssait.is_assignment())
        continue;

      expr2tc new_lhs = ssait.original_lhs;
      renaming::renaming_levelt::get_original_name(
        new_lhs, symbol_renaming_level::level0);

      if (all_loop_vars.find(new_lhs) == all_loop_vars.end())
        continue;

      var_ssa_list[to_symbol2t(new_lhs).thename] = {
        ssait.original_lhs, ssait.rhs};
    }

    if (!var_ssa_list.size())
      return;

    // Query the solver for the value of each variable
    std::vector<expr2tc> equalities;
    for (auto it : var_ssa_list)
    {
      // We don't support arrays or pointers
      if (is_array_type(it.second.first) || is_pointer_type(it.second.first))
        return;

      auto lhs = build_lhs(smt_conv, it.second.first);
      auto value = build_rhs(smt_conv, it.second.second);

      // Add lhs and rhs to the list of new constraints
      equalities.push_back(equality2tc(lhs, value));
    }

    // Build new assertion
    expr2tc constraints = equalities[0];
    for (std::size_t i = 1; i < equalities.size(); ++i)
      constraints = and2tc(constraints, equalities[i]);

    // and add it to the goto program
    goto_programt::targett loop_exit = lit->get_original_loop_exit();

    goto_programt::instructiont i;
    i.make_assertion(not2tc(constraints));
    i.location = loop_exit->location;
    i.location.user_provided(true);
    i.loop_number = loop_exit->loop_number;
    i.inductive_assertion = true;

    fit->second.body.insert_swap(loop_exit, i);

    // recalculate numbers, etc.
    symex->goto_functions.update();
    return;
  }
}

smt_resultt bmct::run_thread(std::shared_ptr<symex_target_equationt> &eq)
{
  // Clear collected pytest test data at the start of coverage run
  if (options.get_bool_option("generate-pytest-testcase"))
    pytest_gen.clear();

  // Clear collected ctest test data at the start of coverage run
  if (options.get_bool_option("generate-ctest-testcase"))
    ctest_gen.clear();

  fine_timet symex_start = current_time();
  try
  {
    goto_symext::symex_resultt solver_result =
      options.get_bool_option("schedule") ? symex->generate_schedule_formula()
                                          : symex->get_next_formula();

    fine_timet symex_stop = current_time();

    eq =
      std::dynamic_pointer_cast<symex_target_equationt>(solver_result.target);

    saw_bounded_loop_truncation |= solver_result.bounded_loop_truncations > 0;

    log_status(
      "Symex completed in: {}s ({} assignments)",
      time2string(symex_stop - symex_start),
      eq->SSA_steps.size());

    if (options.get_bool_option("double-assign-check"))
    {
      const bool ssa_names_unique = eq->check_for_duplicate_assigns();
      SYMEX_INVARIANT(
        ssa_names_unique, "the equation defines an SSA name more than once");
    }

    BigInt ignored;
    for (auto &a : algorithms)
    {
      a->run(eq->SSA_steps);
      ignored += a->ignored();
    }

    // Count remaining assertions after all algorithms have run
    BigInt remaining_asserts = 0;
    for (const auto &step : eq->SSA_steps)
    {
      if (step.is_assert() && !step.ignore)
        ++remaining_asserts;
    }

    seed_property_verdicts(*eq);

    if (
      options.get_bool_option("program-only") ||
      options.get_bool_option("program-too"))
      show_program(*eq);

    if (options.get_bool_option("program-only"))
      return P_SMTLIB;

    log_status(
      "Generated {} VCC(s), {} remaining after simplification ({} "
      "assignments)",
      solver_result.total_claims,
      remaining_asserts,
      BigInt(eq->SSA_steps.size()) - ignored);

    if (options.get_bool_option("show-vcc"))
    {
      show_vcc(*eq);
      return P_SMTLIB;
    }

    if (solver_result.remaining_claims == 0)
    {
      if (options.get_bool_option("smt-formula-only"))
      {
        log_status(
          "No VCC remaining, no SMT formula will be generated for"
          " this program\n");
        return P_SMTLIB;
      }

      // In coverage mode, still print the coverage summary even when all
      // claims are simplified away (e.g., straight-line code with 0 branches).
      //
      // BUT NOT under --k-induction / --incremental-bmc. Those drivers run
      // several bmct phases and emit the authoritative summary themselves
      // (do_bmc_strategy in esbmc_parseoptions.cpp), from the accumulated
      // goto_functionst::reached_claims. A phase whose equation simplifies
      // to zero claims -- which is the normal outcome of the final inductive
      // step -- would otherwise print a second "[Coverage] ... Reached : 0 ...
      // Branch Coverage: 0%" block from the deliberately EMPTY sets below,
      // immediately before the real one. Two blocks, the first always 0%:
      // any consumer that reads the first match (or greps `head -1`)
      // concludes k-induction covered nothing, when the run in fact reached
      // 9/10 on the same input. multi_property_check's own report_coverage
      // call already carries exactly this guard; this early-return path was
      // missing it.
      if (
        options.get_bool_option("multi-property") &&
        !options.get_bool_option("k-induction") &&
        !options.get_bool_option("incremental-bmc"))
      {
        std::unordered_set<std::string> empty_reached;
        std::unordered_multiset<std::string> empty_mul_reached;
        pytest_generator empty_pytest;
        ctest_generator empty_ctest;
        foundry_generator empty_foundry;
        report_coverage(
          options,
          empty_reached,
          empty_mul_reached,
          empty_pytest,
          empty_ctest,
          empty_foundry);
      }

      return P_UNSATISFIABLE;
    }

    if (options.get_bool_option("ltl"))
    {
      int res = ltl_run_thread(*eq);
      if (res == -1)
        return P_SMTLIB;
      if (res == ltl_res_uninstrumented)
      {
        ltl_uninstrumented = true;
        return P_UNSATISFIABLE;
      }
      if (res < 0)
        return P_ERROR;
      // Record that we've seen this outcome; later decide what the least
      // outcome was.
      ltl_results_seen[res]++;
      return P_UNSATISFIABLE;
    }

    if (!options.get_bool_option("smt-during-symex"))
    {
      runtime_solver =
        std::unique_ptr<smt_convt>(create_solver("", ns, options));
    }

    if (
      options.get_bool_option("multi-property") &&
      (options.get_bool_option("base-case") ||
       options.get_bool_option("diagnose-unknown-properties") ||
       (options.get_bool_option("inductive-step") &&
        options.get_bool_option("loop-invariant"))))
      return multi_property_check(
        *eq,
        solver_result.remaining_claims,
        *runtime_solver,
        solver_result.bounded_loop_truncations);

    smt_resultt result = run_decision_procedure(*runtime_solver, *eq);

    // Per-claim vacuity probe in single-property mode: a whole-equation
    // reachability check would silently miss a vacuous claim whenever some
    // *other* claim has a reachable path.
    if (
      result == P_UNSATISFIABLE && options.get_bool_option("check-vacuity") &&
      remaining_asserts > 0)
    {
      log_status(
        "Probing {} claim(s) for vacuous discharge",
        remaining_asserts.to_int64());

      for (size_t i = 1; i <= remaining_asserts.to_uint64(); i++)
      {
        symex_target_equationt vac_eq = *eq;
        claim_slicer keeper(
          i, /*show_slice_info=*/false, /*is_goto_cov=*/false, ns);
        keeper.run(vac_eq.SSA_steps);

        if (!is_vacuity_probe_candidate(keeper.claim_property))
          continue;

        if (check_vacuity(vac_eq) == P_UNSATISFIABLE)
        {
          log_warning(
            "Vacuous discharge: claim '{}' has unsatisfiable path "
            "assumptions; possible causes include an over-constrained loop "
            "invariant, requires clause, or upstream assume.",
            keeper.claim_cstr);
          vacuity_detected = true;
        }
      }
    }

    return result;
  }

  catch (std::string &error_str)
  {
    log_error("{}", error_str);
    return P_ERROR;
  }

  catch (const char *error_str)
  {
    log_error("{}", error_str);
    return P_ERROR;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory\n");
    return P_ERROR;
  }
}

int bmct::ltl_run_thread(symex_target_equationt &equation)
{
  /* LTL checking - first check for whether we have a negative prefix, then
   * the indeterminate ones. */
  // Keys are interned irep_idt, matching SSA_stept::comment, so the
  // comparisons below are dstring identity checks with no per-step
  // string materialisation.
  using Type = std::pair<irep_idt, ltl_res>;
  static const std::array seq = {
    Type{"LTL_BAD", ltl_res_bad},
    Type{"LTL_FAILING", ltl_res_failing},
    Type{"LTL_SUCCEEDING", ltl_res_succeeding},
  };

  auto is_prefix_assert = [](const irep_idt &comment) {
    for (const auto &[which, _] : seq)
      if (comment == which)
        return true;
    return false;
  };

  /* Solve `equation` with only the assertions `keep` selects enabled; the rest
   * become skips and are restored before returning. Yields the solver result
   * and how many assertions were actually left to check. */
  auto solve_only = [&](auto keep) {
    std::vector<symex_target_equationt::SSA_stepst::iterator> masked;
    size_t num_asserts = 0;
    for (auto it = equation.SSA_steps.begin(); it != equation.SSA_steps.end();
         ++it)
      if (it->is_assert())
      {
        if (keep(it->comment))
          num_asserts++;
        else
        {
          masked.push_back(it);
          it->type = symex_targett::SKIP;
        }
      }

    smt_resultt solver_result = P_UNSATISFIABLE;
    std::unique_ptr<smt_convt> smt_conv;
    if (num_asserts != 0)
    {
      smt_conv.reset(create_solver("", ns, options));
      solver_result = run_decision_procedure(*smt_conv, equation);
    }

    for (auto &it : masked)
      it->type = symex_targett::ASSERT;

    return std::make_tuple(solver_result, num_asserts, std::move(smt_conv));
  };

  /* A prefix verdict only describes the program if the monitor ran to
   * completion. Everything that is not a prefix assertion -- the unwinding
   * assertions and libltl2ba's own "Unwind bound ... insufficient" guard
   * included -- is masked out below, so check it first: a violation there
   * means the automaton was truncated and no prefix claim follows (#6547). */
  log_status("Checking LTL monitor preconditions");
  smt_resultt guard_result = std::get<0>(
    solve_only([&](const irep_idt &c) { return !is_prefix_assert(c); }));
  switch (guard_result)
  {
  case P_SATISFIABLE:
    log_warning(
      "LTL monitor preconditions violated, the automaton did not run to "
      "completion; prefix outcome is inconclusive");
    return ltl_res_uninstrumented;
  case P_ERROR:
    return -2;
  case P_SMTLIB:
    return -1;
  case P_UNSATISFIABLE:
    break;
  }

  size_t total_prefix_asserts = 0;
  for (const auto &[which, check] : seq)
  {
    log_status("Checking for {}", which);
    auto [solver_result, num_asserts, smt_conv] =
      solve_only([&](const irep_idt &c) { return c == which; });
    total_prefix_asserts += num_asserts;

    if (num_asserts == 0)
      log_warning("Couldn't find {} assertion", which);
    else if (solver_result == P_SATISFIABLE)
      log_status("Found trace satisfying {}", which);

    switch (solver_result)
    {
    case P_SATISFIABLE:
      // Hand the satisfying solver to the trace machinery: report_trace reads
      // the model out of runtime_solver, which an LTL run otherwise never
      // populates because it returns before the solver is created.
      runtime_solver = std::move(smt_conv);
      return check;
    case P_ERROR:
      return -2;
    case P_SMTLIB:
      return -1;
    case P_UNSATISFIABLE:
      continue;
    }
  }

  /* Every prefix assertion was absent rather than discharged, so this formula
   * carries no monitor instrumentation and says nothing about the property.
   * Reporting the top of the lattice here would claim a proof we never ran. */
  if (total_prefix_asserts == 0)
    return ltl_res_uninstrumented;

  /* Otherwise, we just got a good prefix. */
  return ltl_res_good;
}

/// Look the claim's cone up in the cache. Returns true when the stored proof
/// stands in for solving; `cone_key` and `hit` are set for the store step.
static bool proof_cache_hit(
  proof_cachet *cache,
  bool verify,
  const symex_target_equationt::SSA_stepst &steps,
  std::string &cone_key,
  bool &hit)
{
  if (cache == nullptr)
    return false;
  cone_key = ssa_cone_key_string(steps, fingerprint_modet::srcloc);
  hit = cache->proved(cone_key);
  return hit && !verify;
}

/// Store a fresh proof, or fail the run over a stored one the solver has just
/// refuted. `hit` can only be set here under --proof-cache-verify: a hit
/// short-circuits the solve otherwise.
static void proof_cache_store(
  proof_cachet *cache,
  const std::string &cone_key,
  bool hit,
  smt_resultt solver_result,
  bool is_vacuous,
  const std::string &claim,
  smt_resultt &final_result,
  std::mutex &result_mutex)
{
  if (cache == nullptr)
    return;
  // A vacuous discharge reports Unknown, not Passed, so it is not a proof and
  // must not be stored.
  if (solver_result == P_UNSATISFIABLE && !is_vacuous)
  {
    cache->record(cone_key);
    return;
  }

  if (proof_cache_contradicted(hit, solver_result == P_SATISFIABLE))
  {
    log_error(
      "Proof cache: stored proof of '{}' contradicted by the solver", claim);
    std::lock_guard lock(result_mutex);
    final_result = P_ERROR;
    return;
  }

  // Neither re-proved nor refuted, so --proof-cache-verify checked nothing
  // here and must not be read as having done so.
  if (hit)
    log_warning(
      "Proof cache: stored proof of '{}' could not be re-checked", claim);
}

/// A dead-code probe is advisory and a non-probe claim in a coverage run is
/// not being reported, so neither prints a per-claim solve (issue #4495).
static bool
coverage_silences_claim(bool goto_cov, bool dead_code, const std::string &prop)
{
  return goto_cov && (dead_code || prop != "instrumented assertion");
}

/// A coverage probe: SAT means "this location is reachable". It is not a
/// property, so it must not be reported as one (issue #6387). The claims of a
/// path-coverage query are properties.
static bool is_coverage_goal(bool cov_run, const std::string &prop)
{
  return cov_run && !goto_coveraget::path_cov_query_mode() &&
         prop == "instrumented assertion";
}

/// Which ESBMC this is. Hashing the executable is the fallback when the build
/// ID does not name one build, so it is computed once and held.
static const std::string &proof_cache_identity()
{
  static const std::string identity =
    proof_cache_build_identity(esbmc_build_id());
  return identity;
}

/// Build the cache only where a stored proof would be sound; nullptr disables
/// every cache interaction downstream. A cached proof is only sound where the
/// claim's sliced cone is everything its verdict depends on. The k-induction
/// phases and --incremental-bmc qualify: each sets its phase flags and unwind
/// on this optionst before running, so both are already part of a claim's key.
static std::unique_ptr<proof_cachet> make_proof_cache(
  const optionst &options,
  const std::string &dir,
  const BigInt &interleaving_number)
{
  if (dir.empty())
    return nullptr;

  // Reasons the option set alone decides were reported before any solving, by
  // proof_cache_flags_usable in parseoptions/driver.cpp.
  if (!proof_cache_inactive_reason(options).empty())
    return nullptr;

  if (interleaving_number > 1)
  {
    report_proof_cache_inactive(
      "thread interleavings after the first, where a claim carries a schedule "
      "its cone does not name");
    return nullptr;
  }

  // A proof must not outlive the build that produced it, so a key that cannot
  // name the verifier is no key at all.
  if (proof_cache_identity().empty())
  {
    report_proof_cache_inactive(
      "the running esbmc binary could not be identified");
    return nullptr;
  }

  return std::make_unique<proof_cachet>(dir, options, proof_cache_identity());
}

/// A warm run solves nothing, so without this the report shows no solver
/// activity at all and gives no sign of where the verdicts came from.
static void log_proof_cache_summary(const proof_cachet *cache)
{
  if (cache == nullptr)
    return;
  log_status(
    "Proof cache: {} claim(s) reused, {} solved",
    cache->hits(),
    cache->misses());
}

/// One line per solved claim, digesting its cone under each normalisation:
/// the instrument for measuring what --proof-cache can reuse.
/// No-op when --claim-fingerprint-dump was not given.
static void dump_claim_fingerprint(
  const std::string &path,
  const symex_target_equationt::SSA_stepst &cone,
  smt_resultt result,
  const std::string &loc,
  const std::string &msg)
{
  if (path.empty())
    return;

  const char *verdict = result == P_UNSATISFIABLE ? "UNSAT"
                        : result == P_SATISFIABLE ? "SAT"
                                                  : "OTHER";

  std::string line = fmt::format(
    "{:016x}\t{:016x}\t{:016x}\t{}\t{}\t{}\t{}",
    ssa_cone_digest(cone, fingerprint_modet::raw),
    ssa_cone_digest(cone, fingerprint_modet::srcloc),
    ssa_cone_digest(cone, fingerprint_modet::full),
    ssa_cone_size(cone),
    verdict,
    loc,
    msg);

  static std::mutex dump_mutex;
  std::lock_guard<std::mutex> lock(dump_mutex);
  if (path == "-")
  {
    log_status("CLAIM-FP {}", line);
    return;
  }
  std::ofstream out(path, std::ios::app);
  out << line << "\n";
}

smt_resultt bmct::multi_property_check(
  const symex_target_equationt &eq,
  size_t remaining_claims,
  smt_convt &runtime_solver,
  unsigned int truncated_loops)
{
  // Initial values
  smt_resultt final_result = P_UNSATISFIABLE;
  std::mutex result_mutex;
  std::atomic<size_t> ce_counter{0};
  // How many claims this run has actually DECIDED (a solver verdict came back).
  // Three consumers, and none of them can be served by a count taken after the
  // loop, because the whole point is that the loop may not finish:
  //   * the "mid-solve after claim N of M" label on every incremental publish;
  //   * the PARTIAL report's `claims_decided` / `claims_total`;
  //   * the signal-safe snapshot the kill handler reads.
  std::atomic<size_t> decided_claims{0};
  // Sequential default consumes `jobs` in iteration order; using a sorted
  // vector lets us solve user-source claims before c2goto/library claims so
  // multi-property doesn't burn the budget on spurious library-side dereference
  // / overflow checks ahead of the user's real bugs.
  std::vector<size_t> jobs;

  // For coverage info
  auto &reached_claims = symex->goto_functions.reached_claims;
  auto &reached_mul_claims = symex->goto_functions.reached_mul_claims;
  auto &reached_claims_mutex = symex->goto_functions.reached_claims_mutex;
  auto &reached_mul_claims_mutex =
    symex->goto_functions.reached_mul_claims_mutex;

  if (options.get_bool_option("coverage-measurement"))
  {
    // Per pass, not cumulative: under --k-induction / --incremental-bmc this
    // runs once per phase per k step, and a goal left undecided at k=1 may
    // well be decided at k=2.
    std::lock_guard lock(cov_report_mutex);
    undecided_cov_goals = 0;
    cov_incomplete_reasons.clear();
  }

  // A bounded loop cut off without an unwinding assertion leaves no other
  // trace, and the goals past the bound were never emitted, so a percentage
  // measured here is a lower bound (issue #6387). Taken from the symex result
  // rather than live exploration state, which --schedule has already
  // invalidated by now (issue #6423).
  if (options.get_bool_option("coverage-measurement") && truncated_loops > 0)
    note_cov_incomplete(
      fmt::format(
        "the unwinding bound cut off {} loop iteration(s) with unwinding "
        "assertions disabled, so goals past the bound were never explored",
        truncated_loops));

  // "Assertion Cov"
  bool is_assert_cov = options.get_bool_option("assertion-coverage") ||
                       options.get_bool_option("assertion-coverage-claims");
  // "Condition Cov"
  bool is_cond_cov = options.get_bool_option("condition-coverage") ||
                     options.get_bool_option("condition-coverage-claims") ||
                     options.get_bool_option("condition-coverage-rm") ||
                     options.get_bool_option("condition-coverage-claims-rm");
  // "Branch Cov"
  bool is_branch_cov = options.get_bool_option("branch-coverage") ||
                       options.get_bool_option("branch-coverage-claims");
  bool is_branch_func_cov =
    options.get_bool_option("branch-function-coverage") ||
    options.get_bool_option("branch-function-coverage-claims");
  // "k-Path Cov" — keyed off the dedicated k-path-coverage-enabled
  // boolean (see the note where it is set above); needed in the
  // is_goto_cov disjunction so the claim_slicer reads the witness
  // comment, matching the form stored in goto_coveraget::all_claims.
  bool is_k_path_cov = options.get_bool_option("k-path-coverage-enabled");
  // "Solidity complete-path Cov" — goals carry enc(pi) in the comment, same
  // as k-path, so it must join the is_goto_cov disjunction below or
  // claim_slicer would read the negated-expr instead of the comment and
  // every JSON entry would show up uncovered.
  bool is_path_cov = options.get_bool_option("solidity-path-coverage-enabled");

  // "Dead code" (advisory) reuses the branch-coverage instrumentation, so it
  // needs the same goto-cov claim handling: claim_slicer must read the probe
  // comment and reached_claims must be keyed by "comment\tloc" to match
  // goto_coveraget::all_claims (otherwise every probe looks unreached and
  // every branch is misreported as dead — issue #4495).
  bool is_dead_code = options.get_bool_option("dead-code-check");
  // A coverage *measurement* run. Deliberately excludes --dead-code-check:
  // that mode borrows the same probes but keeps a verdict and reports CWE-561
  // advisories, so none of the coverage reporting rules below apply to it.
  const bool is_cov_run = options.get_bool_option("coverage-measurement");

  // is_vb: enable verbose output coverage info if the option "--verbosity
  // coverage:N" is set, where N should larger than 0 By enabling this, we will
  // output the coverage information when handling each instrumentation
  // assertion.
  bool is_vb = messaget::state.modules["coverage"] != VerbosityLevel::None;

  // For incr/kind in multi-property
  bool is_keep_verified = options.get_bool_option("keep-verified-claims");
  const std::string fingerprint_dump =
    options.get_option("claim-fingerprint-dump");

  // Only where a claim's sliced cone is all its verdict depends on -- the
  // exclusions the in-run assertion_cache makes, plus the coverage probes.
  const std::string proof_cache_dir = options.get_option("proof-cache");
  const bool proof_cache_verify = options.get_bool_option("proof-cache-verify");
  std::unique_ptr<proof_cachet> proof_cache =
    make_proof_cache(options, proof_cache_dir, interleaving_number);
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  const bool withhold_proofs = withholds_proofs(options);
  const bool clears_proved_claims =
    !is_keep_verified && !bs && !withhold_proofs;

  // For multi-fail-fast
  const std::string fail_fast = options.get_option("multi-fail-fast");
  const bool is_fail_fast = !fail_fast.empty() ? true : false;
  const int fail_fast_limit = is_fail_fast ? stoi(fail_fast) : 0;
  std::atomic<int> fail_fast_cnt{0};

  if (is_fail_fast && fail_fast_limit < 0)
  {
    log_error("the value of multi-fail-fast should be positive!");
    abort();
  }

  // TODO: This is the place to check a cache
  jobs.reserve(remaining_claims);
  for (size_t i = 1; i <= remaining_claims; i++)
    jobs.push_back(i);

  // Published BEFORE the first solve, so that a run killed during job 1 still
  // has a denominator to report its partial numerator against. Set here rather
  // than in the pass, because it is the count of claims that survived
  // simplification and reached this loop — which is what "decided so far" is a
  // fraction of, and is not the instrumented path count.
  goto_coveraget::live_decided.store(0, std::memory_order_relaxed);
  goto_coveraget::claims_total_atomic.store(
    remaining_claims, std::memory_order_relaxed);

  // The exact set of claims this loop was given, recorded BEFORE it starts.
  // It is what lets a partial run tell "the loop never got to this claim" from
  // "the simplifier folded this claim away and no loop was ever going to see
  // it" -- two facts with opposite next actions that would otherwise share one
  // U-reason cell. Taken from the equation rather than from the job indices
  // because an index says nothing about WHICH claim it is until the per-job
  // slicer has run, and a job that never runs never slices.
  if (is_path_cov)
  {
    std::set<std::string> queued;
    for (const auto &step : eq.SSA_steps)
      if (step.is_assert() && !step.ignore)
        queued.insert(step.comment.as_string());
    goto_coveraget::claims_in_solve_loop.swap(queued);

    // ---- ARM THE PER-KEY MULTIPLICITY CHECK ----
    //
    // The ceiling is the number of transactions the harness executes, because
    // that is how many times one assert instruction can be reached. The rule
    // that produces it is get_tx_bound() in the Solidity frontend: an explicit
    // --solidity-max-tx wins, and path coverage's default is 2. `0` is NOT
    // unbounded here -- coverage rewrites the dispatcher back-edge to a SKIP,
    // leaving exactly one transaction -- so it maps to 1, not to "no limit".
    //
    // --path-cov-max-claim-solves overrides it, and exists so the REFUSAL can
    // be exercised: with a sound instrumentation the ceiling is never reached,
    // so a check armed only from the transaction count could never be shown to
    // fire, and an unfireable check is the shape this pass has already shipped.
    size_t tx = 2;
    const std::string mt = options.get_option("solidity-max-tx");
    if (!mt.empty())
    {
      const long v = std::stol(mt);
      tx = v <= 0 ? 1 : (size_t)v;
    }
    // ---- THE TRANSACTION COUNT IS NOT THE WHOLE CEILING ----
    //
    // An external call is modelled as nondet RE-ENTRY into the contract's own
    // dispatcher, so one instrumented assert is instantiated once per re-entry
    // level, and the level count is `--unwind`. MEASURED on st1inch, 22 unit
    // logs (notes/coverage/D33): the five units that call out
    // (safeTransferFrom / safeTransfer / Address.sendValue) have a VCC/path
    // ratio of exactly 4.00 against `--unwind 4`, while the fifteen that do not
    // sit at 1.00. That is the documented behaviour of the re-entry model, not
    // duplicate instrumentation, and a ceiling of `tx` alone would abort every
    // one of those units.
    //
    // So the ceiling is tx x unwind: a sound upper bound on how many times one
    // assert INSTRUCTION can be reached. It is deliberately loose. The precise
    // duplication D33 names -- a constructor that calls a public unit, giving
    // the unit's body two identities under one claim key -- sits well inside
    // this bound and is NOT what this check is for; it is addressed by taking
    // the deployment out of the unit query (--path-cov-fixture). What this
    // catches is gross duplication, and what makes the D33 case visible is the
    // superseded-verdict counter below, which is exactly its harm: the reported
    // reason depends on which instantiation was solved first.
    size_t unwind = 4;
    const std::string uw = options.get_option("unwind");
    if (!uw.empty())
    {
      const long v = std::stol(uw);
      if (v > 0)
        unwind = (size_t)v;
    }
    // ---- AND THE BOUNDS ARE STILL NOT A CEILING. NOT ENFORCED BY DEFAULT ----
    //
    // `tx x unwind` was the second attempt and it is also wrong, which is what
    // decided this. MEASURED on
    // regression/esbmc-solidity/solidity_path_cov_residual_unit_call_obstacle
    // at `--solidity-max-tx 1 --unwind 1`, i.e. a derived ceiling of 1: the key
    // `c:path:2` is solved TWICE and the run is legitimate. `c` is a public
    // unit that units `a` and `b` also call internally, and the depth bound
    // leaves those calls unexpanded, so `c`'s body is instantiated once as its
    // own dispatched entry and once inside each caller. That is the recorded
    // unit-body double identity (notes/coverage/D33), it is not a defect of
    // this run, and the count it produces is NOT any of the run's bounds -- it
    // is a property of the call graph and of which calls the depth bound
    // expanded.
    //
    // So a sound ceiling is not derivable from tx and unwind, and a check that
    // aborts on a derived one turns two green regressions red. It is therefore
    // NOT enforced unless the caller states a ceiling explicitly with
    // --path-cov-max-claim-solves: an assertion the caller owns, on a
    // configuration the caller knows the shape of. The derived number is still
    // computed and printed, as context for reading the multiplicity, and is
    // labelled as not enforced so it cannot be mistaken for a bound that held.
    size_t allowed = 0; // 0 = measure, do not refuse
    const size_t derived_ceiling = tx * unwind;
    std::string origin =
      "this run explores " + std::to_string(tx) + " transaction(s) at unwind " +
      std::to_string(unwind) +
      ", but neither bounds how often one assert instruction is instantiated: "
      "a public unit called internally by another unit is instantiated once "
      "per caller as well";
    const std::string cap = options.get_option("path-cov-max-claim-solves");
    if (!cap.empty())
    {
      allowed = (size_t)std::stoul(cap);
      origin = "--path-cov-max-claim-solves set the ceiling to " +
               std::to_string(allowed) +
               "; nothing is enforced without it, because tx x unwind (" +
               std::to_string(derived_ceiling) +
               " here) is not a bound on instantiations";
      log_status(
        "--path-cov-max-claim-solves {}: one claim key may be decided at most "
        "{} time(s) this run, and the run is REFUSED beyond that. Without this "
        "option nothing is enforced: tx {} x unwind {} = {} is context, not a "
        "bound -- a public unit called internally by another unit is "
        "instantiated once per caller too",
        allowed,
        allowed,
        tx,
        unwind,
        derived_ceiling);
    }
    path_cov_allowed_solves = allowed;
    path_cov_allowed_solves_origin = origin;
  }

  // Reorder so user-source claims solve before c2goto/library claims. Walk
  // SSA_steps once, mapping each assertion's 1-based index to a bool flag
  // "is in user source". Library paths contain "c2goto/library" or "/library/";
  // anything else (the user's input source file) is user-side. Stable sort
  // preserves intra-bucket order, keeping CE numbering deterministic.
  if (remaining_claims > 0)
  {
    std::vector<bool> is_user(remaining_claims + 1, false);
    size_t counter = 0;
    for (auto const &step : eq.SSA_steps)
    {
      if (!step.is_assert())
        continue;
      ++counter;
      if (counter > remaining_claims)
        break;
      const std::string file = step.source.pc->location.get_file().as_string();
      const bool is_library =
        file.find("c2goto/library") != std::string::npos ||
        file.find("/library/") != std::string::npos;
      is_user[counter] = !is_library;
    }
    std::stable_sort(jobs.begin(), jobs.end(), [&is_user](size_t a, size_t b) {
      return is_user[a] && !is_user[b];
    });
  }

  /* This is a JOB that will:
   * 1. Generate a solver instance for a specific claim (@parameter i)
   * 2. Solve the instance
   * 3. Generate a Counter-Example (or Witness)
   *
   * This job also affects the environment by using:
   * - &ce_counter: for generating the Counter Example file name
   * - &final_result: if the current instance is SAT, then we known that the
   * current k contains a bug
   *
   * Finally, this function is affected by the "multi-fail-fast" option, which
   * makes this instance stop if final_result is set to SAT
   */
  // FAULT INJECTION, shipped rather than kept in a throwaway build.
  //
  // The three mechanisms this file now carries -- the mid-solve payload
  // publish, the PARTIAL report on the exception path, and the signal arm --
  // all only fire on a run that does NOT reach a clean exit. A regression that
  // cannot produce such a run cannot pin any of them, and `test.desc` describes
  // exactly one invocation with no environment of its own and with --timeout /
  // --memlimit stripped by the harness. So the fault has to be reachable from
  // the command line or it is not testable at all, and an untested rescue path
  // is the shape this project has already shipped twice (a function written and
  // never called; a guard that was always true).
  //
  // Both are 0 (off) unless asked for, and both are refused outside
  // --solidity-path-coverage so no ordinary run can trip over them.
  const size_t fault_after =
    is_path_cov && !options.get_option("path-cov-fault-after").empty()
      ? (size_t)std::stoul(options.get_option("path-cov-fault-after"))
      : 0;
  const size_t fault_sigterm =
    is_path_cov && !options.get_option("path-cov-fault-sigterm").empty()
      ? (size_t)std::stoul(options.get_option("path-cov-fault-sigterm"))
      : 0;
  // Throws from INSIDE the counterexample harvest -- after the verdict is
  // recorded in claim_outcome and before the claim signature reaches
  // reached_claims. The two records of one refutation are written at different
  // times and everything between them can throw, so that window needs a fault
  // of its own: `fault_after` fires at the START of a job, by which point every
  // earlier claim has completed all of its side effects, and can therefore
  // never produce the state P16_Mapping produced.
  const size_t fault_mid_witness =
    is_path_cov && !options.get_option("path-cov-fault-mid-witness").empty()
      ? (size_t)std::stoul(options.get_option("path-cov-fault-mid-witness"))
      : 0;

  // Jobs skipped because their claim key was already witnessed.
  std::atomic<size_t> skipped_witnessed_copies{0};

  auto job_function = [this,
                       &eq,
                       &ce_counter,
                       &decided_claims,
                       &remaining_claims,
                       &fault_after,
                       &fault_sigterm,
                       &fault_mid_witness,
                       &final_result,
                       &result_mutex,
                       &reached_claims,
                       &reached_mul_claims,
                       &reached_claims_mutex,
                       &reached_mul_claims_mutex,
                       &is_assert_cov,
                       &is_cond_cov,
                       &is_vb,
                       &is_branch_cov,
                       &is_branch_func_cov,
                       &is_k_path_cov,
                       &is_path_cov,
                       &is_dead_code,
                       &is_cov_run,
                       &is_keep_verified,
                       &skipped_witnessed_copies,
                       &fingerprint_dump,
                       &proof_cache,
                       &proof_cache_verify,
                       &is_fail_fast,
                       &fail_fast_limit,
                       &fail_fast_cnt,
                       &bs,
                       &fc,
                       &is,
                       &withhold_proofs,
                       &clears_proved_claims,
                       &runtime_solver](const size_t &i) {
    // Fault injection (see the two constants above). Checked BEFORE this job
    // does anything, so the N claims already decided have completed every
    // side effect they own -- including the incremental covered-set publish,
    // which is the thing the step-1 regression is asserting survived.
    if (fault_after && decided_claims.load() >= fault_after)
    {
      log_error(
        "--path-cov-fault-after {}: injecting std::bad_alloc after {} decided "
        "claim(s) (fault injection; this is not a real allocation failure)",
        fault_after,
        decided_claims.load());
      throw std::bad_alloc();
    }
    if (fault_sigterm && decided_claims.load() >= fault_sigterm)
    {
      log_error(
        "--path-cov-fault-sigterm {}: raising SIGTERM after {} decided "
        "claim(s) (fault injection; this is not a real external kill)",
        fault_sigterm,
        decided_claims.load());
      raise(SIGTERM);
    }

    //"multi-fail-fast n": stop after first n SATs found. A coverage run has
    // to identify the claim first: only instrumented probes count towards the
    // goal tally, so bailing here would make "N goal(s) undecided" disagree
    // with the goal count in the summary line.
    const bool fail_fast_hit = is_fail_fast && fail_fast_cnt >= fail_fast_limit;
    if (fail_fast_hit)
    {
      // The skipped claims reach no verdict, so the report is a partial view
      // of the program's properties and must say so.
      report_incomplete = true;
      if (!is_cov_run)
        return;
    }

    // Since this is just a copy, we probably don't need a lock
    symex_target_equationt local_eq = eq;

    // Set up the current claim and disable slice info output.
    // `is_goto_cov` flips claim_slicer's `claim_msg` source: in goto-cov
    // modes the slicer reads the comment (the original witness/guard
    // text we stored in insert_assert); otherwise it reads the negated
    // assertion expression. k-path goals are stored the same way as
    // branch / condition goals, so they must be in this disjunction —
    // otherwise the claim_sig built just below disagrees with the
    // form in goto_coveraget::all_claims and every JSON entry shows up
    // as uncovered even when reached_claims has the matching reached
    // signature (PR #4330 review).
    const bool is_goto_cov = is_cov_run || is_dead_code;
    claim_slicer claim(i, false, is_goto_cov, ns);
    claim.run(local_eq.SSA_steps);

    const property_locationt &claim_ploc = claim.claim_ploc;

    if (fail_fast_hit)
    {
      // The skipped probes were never solved. Counting them as unreached
      // would report a percentage that looks measured but is not.
      if (claim.claim_property == "instrumented assertion")
      {
        goto_functionst::property_verdicts.record(
          claim.claim_key, property_verdictt::Unknown, claim_ploc);
        note_undecided_cov_goal("--multi-fail-fast limit reached");
      }
      return;
    }

    // Skip a claim already found violated. property_key still merges two
    // claims of one kind that symex raises at one instruction, such as the
    // NULL checks of `*p + *q` (discussion #7900).
    bool is_verified = false;
    std::string claim_sig = claim.claim_msg + "\t" + claim.claim_loc;
    if (is_assert_cov)
    {
      // C++20 reached_mul_claims.contains
      std::lock_guard lock(reached_mul_claims_mutex);
      is_verified = reached_mul_claims.count(claim_sig) ? true : false;
    }
    else
    {
      // THE LOOKUP KEY MUST BE THE ONE THE INSERT USES. The insert below
      // discriminates -- `claim_sig` (msg + "\t" + loc) when `is_goto_cov`,
      // `claim_cstr` (msg + " at " + loc) otherwise -- and this lookup did not.
      // So under coverage it asked for a spelling that is never stored,
      // `is_verified` was ALWAYS false, and the skip a few lines down was dead
      // code: every symex instantiation of one instrumented assert was solved
      // again from scratch.
      //
      // MEASURED, EscrowDst.withdraw: 5 distinct claim keys, 425 VCCs, ~85
      // solves per path. All four obtainable witnesses were in hand after the
      // first 46 solves; the remaining 379 would have re-derived the same four
      // and the process ran out of memory at 8 GiB first. Same shape on
      // st1inch `setFeeReceiver` at 2x (10 VCCs for 5 paths). The tool's own
      // `Verdicts Preserved` line has been reporting this all along -- its
      // header calls a non-zero value "itself a defect (duplicate
      // instrumentation)" -- and nothing acted on it.
      //
      // The premise was checked before the change rather than after: every
      // repeated `Solving claim` line for one path is BYTE-IDENTICAL across all
      // its solves, so a lookup on `claim_sig` really does match. If they had
      // differed in their location suffix the repair would have fired zero
      // times while costing a string build per job
      // (notes/coverage/scripts/claim_key_identity.py is that check).
      //
      // SOUND FOR COVERAGE because `reached_claims` only ever receives REFUTED
      // keys (the insert sits inside the P_SATISFIABLE arm), and for coverage F
      // is monotone: once a path has a witness it is covered, and further
      // witnesses of the same path are `--all-witnesses` material, which is
      // enumerated INSIDE one solve, not across repeats. No 'P'/'U'/'B' verdict
      // can reach this skip.
      //
      // SCOPED TO PATH COVERAGE ON PURPOSE. Repairing it for `is_goto_cov`
      // would newly enable skipping under branch/condition/k-path coverage as
      // well, and the branch-coverage dataset this project compares against is
      // LOCKED. Moving its numbers as a side effect of a path-coverage fix is
      // precisely the kind of change that cannot be attributed afterwards.
      std::lock_guard lock(reached_claims_mutex);
      is_verified =
        reached_claims.count(is_path_cov ? claim_sig : claim.claim_key) ? true
                                                                        : false;
    }
    if (is_assert_cov && is_verified)
    {
      // insert to the multiset before skipping the verification process
      std::lock_guard lock(reached_mul_claims_mutex);
      reached_mul_claims.emplace(claim_sig);
    }

    // skip if we have already verified
    if (is_verified && !is_keep_verified)
    {
      ++skipped_witnessed_copies;
      return;
    }

    // Slice
    if (!options.get_bool_option("no-slice"))
    {
      symex_slicet slicer(options);
      slicer.run(local_eq.SSA_steps);
    }

    if (options.get_bool_option("ssa-features-dump"))
    {
      ssa_features features;
      features.run(local_eq.SSA_steps);
    }

    std::string cone_key;
    bool cached_proof = false;
    if (
      proof_cache_hit(
        proof_cache.get(),
        proof_cache_verify,
        local_eq.SSA_steps,
        cone_key,
        cached_proof))
    {
      record_discharge(
        withhold_proofs,
        claim.claim_key,
        property_verdictt::Passed,
        claim_ploc);

      // A reused proof has to leave the run in the state a fresh one would,
      // or a warm k-induction / --incremental-bmc run keeps re-symexing the
      // claims a cold one had already dropped -- the opposite of the point
      // (esbmc/esbmc#7143). Same guard as the P_UNSATISFIABLE arm below.
      if (clears_proved_claims)
      {
        clear_verified_claims_in_ssa(local_eq, claim, is_goto_cov);
        clear_verified_claims_in_goto(claim, is_goto_cov);
      }
      return;
    }

    // Initialize a solver
    smt_convt *solver_ptr = &runtime_solver;
    std::unique_ptr<smt_convt> new_solver;
    if (!options.get_bool_option("smt-during-symex"))
    {
      new_solver = std::unique_ptr<smt_convt>(create_solver("", ns, options));
      solver_ptr = new_solver.get();
    }

    // --smt-during-symex shares one persistent solver across every claim.
    // Scope this claim's re-encoded formula in a context frame; without it
    // the negated assertion stays asserted forever, and once one claim's
    // negation is unsatisfiable every later claim solves UNSAT and is
    // misreported as PASSED (issue #6540).
    struct solver_ctx_framet
    {
      smt_convt *conv;
      explicit solver_ctx_framet(smt_convt *c) : conv(c)
      {
        if (conv)
          conv->push_ctx();
      }
      ~solver_ctx_framet()
      {
        if (conv)
          conv->pop_ctx();
      }
    } ctx_frame(new_solver ? nullptr : solver_ptr);

    // Store solver name initially but not again
    std::call_once(solver_stats.name_flag, [&]() {
      solver_stats.name = solver_ptr->solver_text();
    });
    // WHAT ACTUALLY ENFORCED THE BUDGET, recorded from the solver rather than
    // assumed from the flag. A backend with no per-check limit records nothing,
    // and the empty string is turned into an explicit refusal here instead of
    // being left to read as "enforced": a report carrying `claim_timeout_s:
    // 120` for a run nothing bounded is the exact shape of a guard that never
    // fires while everything looks fine.
    if (is_path_cov && goto_coveraget::claim_budget_seconds > 0)
    {
      const std::string mech = smt_timeout_mechanism();
      goto_coveraget::claim_budget_mechanism =
        mech.empty() ? ("NOT ENFORCED: backend '" + solver_ptr->solver_text() +
                        "' has no per-query time limit, so the budget was "
                        "requested and could not be applied")
                     : mech;
    }
    // In coverage mode, only report instrumented coverage claims
    bool is_cov_silent =
      coverage_silences_claim(is_goto_cov, is_dead_code, claim.claim_property);
    const bool is_cov_goal = is_coverage_goal(is_cov_run, claim.claim_property);

    if (!is_cov_silent)
      log_status(
        "Solving claim '{}' with solver {}",
        prettify_solidity_expr(claim.claim_cstr),
        solver_ptr->solver_text());

    // Save current instance with timing
    fine_timet solve_start = current_time();
    smt_resultt solver_result = run_decision_procedure(*solver_ptr, local_eq);
    fine_timet solve_stop = current_time();

    dump_claim_fingerprint(
      fingerprint_dump,
      local_eq.SSA_steps,
      solver_result,
      claim.claim_loc,
      claim.claim_msg);

    // The claims of a path-coverage query are properties, and the query's
    // RESULT line may end the run before any table is printed, so each verdict
    // is named here.
    if (goto_coveraget::path_cov_query_mode() && !is_cov_silent)
    {
      if (solver_result == P_UNSATISFIABLE)
        log_status("✓ PASSED: '{}'", prettify_solidity_expr(claim.claim_cstr));
      else if (solver_result == P_SATISFIABLE)
        log_status("✗ FAILED: '{}'", prettify_solidity_expr(claim.claim_cstr));
    }

    // After UNSAT, probe whether the path to the kept claim is reachable.
    // UNSAT in vacuity mode means the discharge was vacuous -> UNKNOWN.
    bool is_vacuous = false;
    if (
      solver_result == P_UNSATISFIABLE &&
      options.get_bool_option("check-vacuity") &&
      is_vacuity_probe_candidate(claim.claim_property))
    {
      is_vacuous = (check_vacuity(local_eq) == P_UNSATISFIABLE);
      if (is_vacuous)
        vacuity_detected = true;
    }

    proof_cache_store(
      proof_cache.get(),
      cone_key,
      cached_proof,
      solver_result,
      is_vacuous,
      claim.claim_cstr,
      final_result,
      result_mutex);

    // A claim is re-checked in every thread interleaving, and can be
    // discharged in one schedule while being violated in another. Record the
    // outcome rather than reporting it here, so that report_property_verdicts
    // can state the verdict that dominates across the run exactly once.
    // A coverage probe rides the same table: it is re-solved per interleaving
    // just the same, and report_property_verdicts renders it as reachability
    // rather than a verdict, because it is not a property of the program
    // (issue #6387).
    if (!is_cov_silent)
    {
      if (solver_result == P_UNSATISFIABLE)
        record_discharge(
          withhold_proofs,
          claim.claim_key,
          is_vacuous ? property_verdictt::Unknown : property_verdictt::Passed,
          claim_ploc,
          is_vacuous ? "vacuous discharge: path assumptions are unsatisfiable; "
                       "possible causes include an over-constrained loop "
                       "invariant, requires clause, or upstream assume"
                     : "");
      else if (solver_result == P_SATISFIABLE)
        record_satisfiable_claim(claim, claim_ploc, is, local_eq);
      else
      {
        // No answer at all. A coverage run suppresses the verdict that would
        // have reported this; a plain multi-property run reports it nowhere,
        // and a SAT claim elsewhere buries it entirely — so name the claim
        // either way (issue #5934).
        if (solver_result == P_ERROR)
          log_error(
            "SMT solver failed on '{}'",
            prettify_solidity_expr(claim.claim_cstr));
        if (is_cov_goal)
        {
          // Neither reached nor unreached. Recorded so the goal still gets a
          // line and the run closes as INCOMPLETE.
          goto_functionst::property_verdicts.record(
            claim.claim_key, property_verdictt::Unknown, claim_ploc);
          note_undecided_cov_goal(
            solver_result == P_SMTLIB
              ? "SMT formula only, no solving performed"
              : "the solver failed on at least one goal");
        }
      }
    }
    else if (is_goto_cov && solver_result == P_SATISFIABLE)
    {
      // A violated claim the coverage pass did not instrument: another
      // function under --function, or a check symex injects afterwards. A
      // coverage run reports no violations, so without this it would vanish
      // entirely. It is not a completeness problem — whether it truncates
      // exploration depends on the claim — so it is listed separately from
      // the reasons the percentages may be lower bounds.
      note_cov_suppressed_violation(claim.claim_cstr);
    }

    // ---- DID THIS CLAIM BLOW ITS BUDGET? ----
    //
    // The budget is enforced by the SOLVER (a native per-check-sat limit; see
    // solvers/solve.h), and every backend folds "unknown" into P_ERROR --
    // cvc5_conv.cpp maps isUnknown() straight onto it, and smt_resultt
    // has no P_UNKNOWN to fold into instead. So the result alone cannot tell an
    // abandoned query from a genuine solver failure, and the wall clock is what
    // separates them.
    //
    // THE TEST IS UNCONDITIONAL ON WHAT THE SOLVER SAID BEYOND THAT: a SAT or
    // UNSAT answer is kept even if it arrived late, because the solver ANSWERED
    // and an answer is not something to throw away over a stopwatch. Only a
    // non-answer that took at least the budget is filed as abandoned.
    //
    // The 100 ms slack is for clock granularity and for the difference between
    // when the solver's own timer starts and when this one does; without it a
    // limit that fires at exactly the budget can measure as a hair under it and
    // be mislabelled `solver-unknown`, which is the quieter and more misleading
    // of the two errors.
    const bool budget_on = goto_coveraget::claim_budget_seconds > 0;
    const bool answered =
      solver_result == P_SATISFIABLE || solver_result == P_UNSATISFIABLE;
    // ---- THE ELAPSED TIME IS ALREADY IN MILLISECONDS ----
    //
    // `current_time()` returns `tv.tv_usec / 1000 + tv.tv_sec * 1000`
    // (util/time_stopping.cpp), i.e. MILLISECONDS. The old test multiplied that
    // difference by 1000 before comparing it against `budget_seconds * 1000`,
    // so it expanded to
    //
    //     ms * 1000 + 100 >= 120 * 1000      <=>      ms >= 119.9
    //
    // and fired at 120 MILLISECONDS instead of 120 seconds -- a factor of 1000.
    // Every unanswered query slower than about a tenth of a second was filed
    // `claim-budget-exceeded`, and the budget never actually gave a query the
    // time it advertised.
    //
    // MEASURED on st1inch `--focus-function setFeeReceiver --z3
    // --tuple-node-flattener`, whose own log contradicts itself on consecutive
    // lines:
    //
    //     Runtime decision procedure: 5.021s
    //     WARNING: claim budget exceeded (120s): ABANDONING
    //              'setFeeReceiver:path:15 at' ...
    //
    // Five seconds reported as over a two-minute budget. The consequence is not
    // only a wrong number: the five paths were filed under
    // `claim-budget-exceeded` -- "we abandoned it, nothing is known" -- when
    // the solver had ANSWERED, with `unknown`, in seconds. That is the
    // `solver-unknown` cell, and the two call for different next actions
    // (raise the budget vs change the encoding). The token that exists to stop
    // a U from absorbing an unexplained remainder was itself absorbing one.
    //
    // The 100 ms slack keeps its stated meaning now that both sides are in the
    // same unit: it covers clock granularity and the gap between the solver's
    // own timer starting and this one.
    const bool over_budget =
      is_path_cov && budget_on && !answered &&
      (double)(solve_stop - solve_start) + 100.0 >=
        (double)goto_coveraget::claim_budget_seconds * 1000.0;
    if (over_budget)
    {
      goto_coveraget::claim_budget_exceeded.fetch_add(
        1, std::memory_order_relaxed);
      log_warning(
        "claim budget exceeded ({}s): ABANDONING '{}' and continuing to the "
        "next claim. Nothing is known about this path -- it is reported U with "
        "reason 'claim-budget-exceeded', which is NOT 'solver-unknown' (the "
        "solver answering 'I do not know') and NOT 'bounded-holds' (it "
        "answering 'no witness'). Raise --path-cov-claim-timeout to spend more "
        "on it",
        goto_coveraget::claim_budget_seconds,
        prettify_solidity_expr(claim.claim_cstr));
    }

    // Tri-state ledger for Solidity complete-path coverage: record THIS
    // claim's verdict so the report can tell "proven at this bound" apart
    // from "could not decide" (reached_claims records only the refuted
    // ones, so both non-refuted cases would otherwise collapse into a
    // single indistinguishable "uncovered"). An inductive-step SAT means
    // "cannot prove", not a counterexample, so it maps to 'U'.
    if (is_path_cov)
    {
      char verdict;
      if (over_budget)
        // Its own verdict value, not a shade of 'U'. path_u_reason_token's
        // `default` arm returns "" so the caller hard-fails on an unknown
        // verdict, which is exactly why a new one gets its own `case` there
        // rather than being folded into an existing letter.
        verdict = 'B';
      else if (solver_result == P_SATISFIABLE)
        verdict = is ? 'U' : 'F';
      else if (solver_result == P_UNSATISFIABLE)
        verdict = 'P';
      else
        verdict = 'U';
      std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
      // ---- A DECIDED VERDICT IS NEVER REPLACED BY A NON-DECISION ----
      //
      // This used to protect 'F' alone -- "a witness stays valid, so a later
      // phase's failure to reprove must never downgrade it" -- and left every
      // other cell last-writer-wins. 'P' needs exactly the same protection, and
      // for exactly the same reason: it is a DECISION (the solver answered
      // UNSAT), while 'U' and 'B' are the absence of one.
      //
      // MEASURED on st1inch `--focus-function setFeeReceiver --z3
      // --tuple-node-flattener`, one run, and it is not a corner case. The run
      // generates 10 VCCs for 5 paths and solves EACH PATH CLAIM TWICE under
      // the SAME (comment, location) key -- `paths_total` is 5, so
      // `all_claims`, a std::set of that pair, holds five:
      //
      //     path:13   0.010s   ✓ PASSED        then again  2.246s   no verdict
      //     path:12   0.010s   ✓ PASSED        then again  2.054s   no verdict
      //
      // Both were PROVEN in ten milliseconds and both are reported
      // `solver-unknown`. So an unknown share of st1inch's 59 `solver-unknown`
      // -- the number the branch gate turns into a 0 that reads as zero
      // coverage -- is not the solver failing to decide, it is this line
      // discarding a decision it had.
      //
      // 'P' -> 'F' IS STILL ALLOWED, and must be: a claim that held at one
      // exploration can be refuted at a deeper one, and that refutation is a
      // later, stronger DECISION. What is refused is 'P' -> 'U' and 'P' -> 'B',
      // which replace an answer with the absence of an answer.
      //
      // The duplicate solve itself is a SECOND defect and is not fixed here:
      // the same claim key is instrumented at more than one site, so half the
      // solving time on this unit buys nothing. Fixing the bookkeeping makes
      // the numbers right; fixing the duplication also makes them cheaper. They
      // are separate, and folding them into one change would leave neither
      // testable on its own.
      // ---- HOW MANY TIMES HAS THIS KEY BEEN DECIDED? ----
      //
      // Counted before the verdict is merged, so the count is of SOLVES and not
      // of changes: a key solved three times with the same answer each time is
      // still duplicate work, and a rule that only noticed disagreements would
      // report it as clean.
      //
      // The ceiling is the transaction bound: the transaction body is emitted
      // once per transaction, so one assert instruction is reached at most once
      // per transaction. Exceeding it means the claim is instrumented at more
      // than one site -- the duplicate instrumentation the comment below calls
      // a second defect -- and that is fatal rather than counted, because every
      // number this run publishes about that path is then the result of several
      // independent chances to witness it.
      {
        std::lock_guard lk(path_cov_solve_count_mutex);
        const size_t n = ++path_cov_solve_count[claim_sig];
        if (n > 1)
          path_cov_extra_solves.fetch_add(1, std::memory_order_relaxed);
        if (n > path_cov_max_solves)
        {
          path_cov_max_solves = n;
          path_cov_max_solves_key = claim_sig;
        }
        if (path_cov_allowed_solves > 0 && n > path_cov_allowed_solves)
        {
          log_error(
            "--solidity-path-coverage: INTERNAL DEFECT — claim key '{}' has "
            "now been handed to the solver {} times, and at most {} is "
            "allowed ({}). One assert instruction is reached at most once "
            "per transaction, so a higher count means the SAME claim is "
            "instrumented at more than one site. That is not a cost problem: "
            "the path gets several independent chances to be witnessed and "
            "every figure published about it is the result of all of them. "
            "Refusing rather than reporting, because a coverage number "
            "computed from duplicated claims cannot be corrected afterwards",
            prettify_solidity_expr(claim_sig),
            n,
            path_cov_allowed_solves,
            path_cov_allowed_solves_origin);
          abort();
        }
      }

      auto it_o = goto_coveraget::claim_outcome.find(claim_sig);
      if (it_o == goto_coveraget::claim_outcome.end())
        goto_coveraget::claim_outcome.emplace(claim_sig, verdict);
      else if (it_o->second == 'F')
      {
        // final, keep
      }
      else if (it_o->second == 'P' && verdict != 'F')
      {
        // A decision is already recorded and the new outcome is not one.
        // Counted rather than kept silently: "the guard fired" and "the guard
        // was never needed" must be distinguishable, and on this corpus the
        // count is the size of a number already published.
        goto_coveraget::verdicts_preserved.fetch_add(
          1, std::memory_order_relaxed);
      }
      else
      {
        // A DECIDED verdict is being replaced by a DIFFERENT decided one -- in
        // practice only P -> F, the legitimate upgrade: this path holds in one
        // transaction and is refuted in a later one, and the path's answer is
        // the disjunction. Counted here because it is precisely the event
        // `Verdicts Preserved` cannot observe (that counter fires only when the
        // later solve returns NO verdict), and because a silent overwrite and a
        // reasoned upgrade look identical from outside.
        if (
          (it_o->second == 'P' || it_o->second == 'F') &&
          verdict != it_o->second)
          path_cov_verdict_upgrades.fetch_add(1, std::memory_order_relaxed);
        it_o->second = verdict;
      }
    }

    // This claim now HAS a verdict. Counted here rather than at the bottom of
    // the job so that a job which throws while building its counterexample
    // still counts the decision it genuinely made -- the whole quarrel with the
    // old behaviour is that decided work was thrown away.
    const size_t decided_now = ++decided_claims;
    goto_coveraget::live_decided.store(decided_now, std::memory_order_relaxed);

    solver_stats.total_time_ms.fetch_add(solve_stop - solve_start);

    // A claim that reached no verdict — a backend failure (P_ERROR) or an
    // SMTLIB-only emission (P_SMTLIB) — would otherwise leave final_result at
    // its P_UNSATISFIABLE seed, which reads as "every claim discharged" and
    // closes the run SUCCESSFUL over an analysis that never happened. Surface
    // it instead; P_SATISFIABLE still wins, a witnessed violation being a
    // verdict either way (issue #5934).
    if (solver_result == P_ERROR || solver_result == P_SMTLIB)
    {
      // Set even when a SAT claim dominates below: the verdict is right in
      // that case, but the summary is still short a claim and nothing else
      // would say so.
      report_incomplete = true;
      std::lock_guard lock(result_mutex);
      if (final_result != P_SATISFIABLE)
        final_result = solver_result;
    }

    // If an assertion instance is verified to be violated
    if (solver_result == P_SATISFIABLE)
    {
      // Inductive step SAT means unprovable (UNKNOWN), not a real
      // counterexample — skip trace generation and return early.
      if (is)
      {
        if (!is_goto_cov)
        {
          std::lock_guard lock(result_mutex);
          final_result = solver_result;
        }
        return;
      }

      // ---- THE CHAIN REJECTS THIS COUNTEREXAMPLE: RE-SOLVE ONCE ----
      //
      // MEASURED, and it is why this exists: `D10_WrapNotPanic.add` is
      // `require(amt > 0); bal += amt;` and the witness comes back
      // `amt = 2^256-1`, `bal: 500 -> 499`, classified `exit_kind: normal`, and
      // emitted as a bare call asserting a normal exit. `forge test` answers
      // `[FAIL: panic: arithmetic underflow or overflow (0x11)]`. The `require`
      // is load-bearing -- without it the solver picks 0 and the test is green
      // -- so the defect is not "the solver likes extremes": NOTHING in the
      // formula distinguishes a wrapping member of the path's domain from a
      // non-wrapping one, and once the cheap value is excluded the choice is
      // unconstrained.
      //
      // WHY THIS ORDER (solve as before, then re-solve only a WITNESSED path).
      // The alternative -- assume no-overflow on the FIRST solve -- pays one
      // extra query on every path that comes back UNSAT, which on st1inch is
      // the 69 `bounded-holds`. Witnessed paths are single digits per unit, so
      // paying there is the cheap side. It also needs no way to ask "did this
      // model wrap": preferring a non-wrapping witness whenever one exists is
      // strictly stronger than detecting the wrap first, and it removes the
      // only step that would have re-derived arithmetic OUTSIDE the model --
      // a second implementation of an arithmetic the model already performs is
      // free to disagree with it, which is the geometric ladder's own wrap
      // defect.
      //
      // THE PATH CONSTRAINT IS CARRIED FOR FREE, and that closes the hole in
      // the original proposal. The query IS this path's claim
      // `assert(!(tr == enc && cnt == depth))`, so every SAT model is already
      // on this path and adding a conjunct cannot move it to another one. The
      // hole was real, but it belonged to the `assume(no overflow);
      // assert(false)` FORM of the re-solve, not to this architecture.
      // Consequently the UNSAT of this query is exactly the proof that the
      // path is reachable ONLY by overflowing -- free, from the same query.
      std::unique_ptr<smt_convt> arith_solver;
      std::unique_ptr<symex_target_equationt> arith_eq;
      if (
        is_path_cov && options.get_bool_option("path-cov-arith-resolve") &&
        !options.get_bool_option("smt-during-symex") &&
        claim.claim_property == "instrumented assertion")
      {
        arith_eq = std::make_unique<symex_target_equationt>(eq);
        claim_slicer rclaim(i, false, is_goto_cov, ns);
        rclaim.run(arith_eq->SSA_steps);

        // WHICH STEPS, decided on the MACHINE FIELD goto_check sets, never on
        // the comment prose. `add_guarded_claim` stamps
        // `location.property()`, and two different producers share the value
        // `overflow` -- `overflow_check` writes "arithmetic overflow on ..."
        // while `cast_overflow_check` writes "Narrowing cast overflow on ..."
        // and is ON BY DEFAULT for Solidity. A text match would have to know
        // both sentences and would silently miss the next one.
        //
        // THE GUARD HAS TO BE FOLDED IN BY HAND. `convert_internal_step`
        // (symex_target_equation.cpp) uses an ASSUME's `cond` and IGNORES its
        // `guard` -- assumes join the assumption chain unconditionally. So
        // converting the assert as-is would assume "no overflow" even on symex
        // branches where the operation never executes, which can exclude a
        // legitimate witness of this path and report a FALSE
        // "reachable only by overflowing". `implies(guard, cond)` is what
        // makes the assumption say what the assert said.
        size_t n_arith = 0;
        for (auto &st : arith_eq->SSA_steps)
        {
          if (!st.is_assert() || !st.source.is_set)
            continue;
          const std::string prop =
            st.source.pc->location.property().as_string();
          if (prop != "overflow" && prop != "division-by-zero")
            continue;
          st.type = symex_targett::ASSUME;
          st.cond =
            is_nil_expr(st.guard) ? st.cond : implies2tc(st.guard, st.cond);
          st.ignore = false;
          ++n_arith;
        }
        goto_coveraget::arith_conditions_seen.fetch_add(
          n_arith, std::memory_order_relaxed);

        // ⚠ WHAT THIS DELIBERATELY DOES NOT RECORD, and what it costs later.
        //
        // Every matching condition of BOTH families is assumed together and ONE
        // query is issued, so an UNSAT is a statement about the CONJUNCTION:
        // "this path, with every overflow AND every division-by-zero condition
        // holding, is unsatisfiable". It does not say which family -- let alone
        // which operation -- is responsible, and only the path key is stored
        // below (`arith_revert_only_paths` is a bare set, unlike
        // `named_obstacle_paths`, which is a map to a reason).
        //
        // That is enough to REFUSE a case, which is all this does today. It is
        // NOT enough to RENDER one: `vm.expectRevert(stdError.arithmeticError)`
        // asserts Panic 0x11 exactly, and forge matches the revert data byte
        // for byte, so emitting the wrong code is a RED test. Whoever adds the
        // rendering needs the family, and the cheap way to get it is to record
        // the SET of `property()` values actually converted on THIS claim: when
        // that set is a single family the UNSAT is attributable to it by
        // construction, with no extra query. When it holds both, one extra
        // solve per family decides it, and when both of those are UNSAT the
        // firing Panic depends on execution ORDER, which no verdict supplies --
        // there the honest rendering is a bare `vm.expectRevert()`, which
        // asserts exactly what was proved and nothing more.
        //
        // `unchecked { }` is NOT a hazard here, and that is READ rather than
        // assumed: goto_check.cpp's `overflow_check` returns early on
        // `loc.get("#sol_unchecked") == "1"`, so no overflow claim is emitted
        // inside an unchecked block and there is nothing for this loop to
        // convert. P18_Unchecked measured the same thing from the other side --
        // an unchecked block and a normal one produce a byte-identical model.

        // Nothing to assume => nothing to re-solve. Not an error: a unit with
        // no checked operation on this path is the common case, and paying a
        // query to learn that would make the mechanism cost something on every
        // contract instead of on the ones it helps.
        if (n_arith > 0)
        {
          // The conversion must happen BEFORE this slice. symex_slicet::run
          // skips `ignore`d steps outright, so an arithmetic condition left as
          // an ignored ASSERT contributes no symbols to `depends` and the
          // assignments defining its OPERANDS are then free to be sliced away
          // -- after which asserting it would reference symbols the formula no
          // longer constrains and the solver would satisfy it by choosing
          // them. An answer about nothing.
          if (!options.get_bool_option("no-slice"))
          {
            symex_slicet rslicer(options);
            rslicer.run(arith_eq->SSA_steps);
          }
          arith_solver =
            std::unique_ptr<smt_convt>(create_solver("", ns, options));
          const fine_timet ra_start = current_time();
          const smt_resultt ra =
            run_decision_procedure(*arith_solver, *arith_eq);
          const fine_timet ra_stop = current_time();
          goto_coveraget::arith_resolve_queries.fetch_add(
            1, std::memory_order_relaxed);
          goto_coveraget::arith_resolve_ms.fetch_add(
            (size_t)(ra_stop - ra_start), std::memory_order_relaxed);

          if (ra == P_SATISFIABLE)
          {
            // A witness of THIS path that satisfies every enabled arithmetic
            // check. Everything downstream -- the trace, the payload harvest,
            // the Foundry case -- is built from this model instead.
            // SWAP the step lists rather than assign the equation:
            // symex_target_equationt holds `const namespacet &`, so its copy
            // ASSIGNMENT is implicitly deleted (copy construction, used above,
            // is fine). Both equations name the same namespace, so exchanging
            // the only member that differs is exactly the intended effect and
            // is O(1).
            local_eq.SSA_steps.swap(arith_eq->SSA_steps);
            solver_ptr = arith_solver.get();
            goto_coveraget::arith_resolve_replaced.fetch_add(
              1, std::memory_order_relaxed);
            log_status(
              "--path-cov-arith-resolve: '{}' had a counterexample the chain "
              "rejects; re-solved under {} arithmetic condition(s) and took "
              "the non-wrapping witness instead",
              prettify_solidity_expr(claim.claim_msg),
              n_arith);
          }
          else if (ra == P_UNSATISFIABLE)
          {
            // NOT a failure of the re-solve. It is a PROOF: no input reaches
            // this path without violating a checked operation, so on chain the
            // path is reachable only through a Panic revert. Its own cell, and
            // never folded into U -- a U says "we could not decide" and this
            // was decided.
            {
              std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
              goto_coveraget::arith_revert_only_paths.emplace(
                claim.claim_msg, claim.claim_loc);
            }
            log_status(
              "--path-cov-arith-resolve: '{}' is reachable ONLY by violating a "
              "checked arithmetic operation -- the re-solve under {} "
              "condition(s) is UNSAT, which PROVES it. On chain this path is "
              "reached through a Panic revert, so its witness is kept but the "
              "path must not be emitted as a normal-exit test",
              prettify_solidity_expr(claim.claim_msg),
              n_arith);
          }
          else
            log_warning(
              "--path-cov-arith-resolve: the re-solve of '{}' returned neither "
              "sat nor unsat, so the original (possibly chain-rejected) "
              "witness is kept unchanged. This is NOT a proof that the path "
              "needs an overflow",
              prettify_solidity_expr(claim.claim_msg));
        }
      }

      // The path-coverage counterexample payload is harvested from this
      // trace, and the EVM environment (msg./tx./block.) and the harness-side
      // writes are assigned in steps flagged hidden, which a trace otherwise
      // drops. Keeping them changes nothing about what is solved.
      const bool keep_hidden_steps =
        is_path_cov && options.get_bool_option("cov-report-json");

      // --all-witnesses: re-solve with blocking clauses on the nondet input
      // tuple to enumerate further violating inputs at the current k.
      // No re-encoding: we only push extra assertions onto the live solver.
      const bool enumerate = options.get_bool_option("all-witnesses");
      size_t max_w = 1;
      if (enumerate)
      {
        const std::string mw = options.get_option("max-witnesses");
        const int mw_val = mw.empty() ? 16 : std::stoi(mw);
        // 0 means unlimited (only meaningful with --all-witnesses).
        max_w = (mw_val == 0) ? SIZE_MAX : (size_t)mw_val;
      }

      std::vector<witness_recordt> witnesses;
      // ---- ONE CE PAYLOAD PER WITNESS, NOT ONE PER CLAIM ----
      //
      // `--all-witnesses` was fully wired for path coverage and reached every
      // consumer EXCEPT the one this project reads. Per witness the loop below
      // already emitted a `--cex-output` file, a GraphML/YAML witness, a
      // testcase XML, an HTML/JSON report, a pytest/ctest case and a FOUNDRY
      // case -- and then harvested the counterexample payload for the FIRST
      // witness only, discarding the rest. So Foundry got N tests per path and
      // cov-report.json got one set of inputs, which is exactly backwards for a
      // stage-2 ladder that consumes the report.
      //
      // The extra witnesses are also nearly free: they cost N-1 further
      // dec_solve() calls on ONE already-encoded solver instance (one push_ctx,
      // a blocking clause per witness, one pop_ctx) -- no second encoding, no
      // second symex, no second per-claim slice.
      std::vector<goto_coveraget::path_ce_t> ce_all;
      enumeration_stop_reasont stop_reason =
        enumerate ? enumeration_stop_reasont::Unsat
                  : enumeration_stop_reasont::Disabled;

      // Cache option lookups so the per-witness loop body is cheap.
      // A coverage run reports no violations, so it emits no violation
      // artifact for any of its claims: the SV-COMP witness formats can only
      // say "this program violates its specification", and the HTML / JSON
      // reports are violation reports, so either would fabricate a defect.
      // The textual trace and the test-input generators stay on for coverage
      // goals — which values drive execution to a goal is exactly what a
      // coverage run is asked for — rendered as reachability evidence
      // (issue #6387).
      const std::string cex_output =
        (is_cov_goal || !is_goto_cov) ? options.get_option("cex-output") : "";
      const std::string graphml_path =
        is_goto_cov ? "" : options.get_option("witness-output-graphml");
      const std::string yaml_path =
        is_goto_cov ? "" : options.get_option("witness-output-yaml");
      const bool want_graphml = !graphml_path.empty();
      const bool want_yaml = !yaml_path.empty();
      const bool want_testcase = options.get_bool_option("generate-testcase");
      const bool want_html =
        !is_goto_cov && options.get_bool_option("generate-html-report");
      const bool want_json =
        !is_goto_cov && options.get_bool_option("generate-json-report");
      const bool want_pytest =
        options.get_bool_option("generate-pytest-testcase");
      const bool want_ctest =
        options.get_bool_option("generate-ctest-testcase");
      const bool want_foundry =
        options.get_bool_option("generate-foundry-testcase");

      // A bare "{index}-" prefix collides across k-induction phases/k-steps,
      // since ce_counter restarts at zero on every bmct (discussion #6070);
      // tag with phase and k too. Inductive-step and
      // diagnose runs return early at the `if (is) return` guard above, so
      // the ternary only needs base/fwd/bmc.
      const std::string run_phase = bs ? "base" : (fc ? "fwd" : "bmc");
      std::string run_kval = options.get_option("unwind");
      if (run_kval.empty())
        run_kval = "0";

      // Emit testcase metadata once per claim (not once per witness).
      if (want_testcase)
        generate_testcase_metadata();

      // Drive enumeration with a separate variable so the original SAT
      // outcome stays in `solver_result` for downstream bookkeeping
      // (final_result, fail-fast counter, claim cleanup).
      smt_resultt enum_result = solver_result;
      bool ctx_pushed = false;
      while (enum_result == P_SATISFIABLE)
      {
        witness_recordt w;
        build_goto_trace(local_eq, *solver_ptr, w.trace, keep_hidden_steps);
        // Collecting nondet values walks every SSA step and queries the
        // solver model per nondet symbol — non-trivial on coverage runs
        // with many claims and large arrays. Skip it when we don't need
        // it: the values are only consumed by `make_blocking_expr` (only
        // when enumerating) and by the multi-witness pretty-printer
        // (only when --all-witnesses is set, i.e. enumerate==true).
        // The legacy single-witness renderer does not use them.
        if (enumerate)
          w.nondet_inputs = collect_nondet_values(local_eq, *solver_ptr);
        w.ce_index = ce_counter++;

        const std::string witness_id =
          fmt::format("{}-k{}-{}", run_phase, run_kval, w.ce_index);

        // Prefix only the basename, keeping any directory the user gave
        // (e.g. "cex/out" -> "cex/{id}-out").
        auto tag_artifact = [&witness_id](const std::string &path) {
          std::filesystem::path p(path);
          return (p.parent_path() / (witness_id + "-" + p.filename().string()))
            .string();
        };

        // Emit machine-readable artifacts NOW, while this witness's solver
        // model is still live. After the next dec_solve(), the model is
        // either gone (UNSAT) or replaced by the next witness's values.
        if (!cex_output.empty())
        {
          std::ofstream out(tag_artifact(cex_output));
          show_goto_trace(out, ns, w.trace, is_cov_goal);
        }
        // For graphml/yaml the writer reads the path from `options`;
        // override per-witness so multiple witnesses don't overwrite the
        // same file (and so it's safe under --parallel-solving).
        if (want_graphml)
          violation_graphml_goto_trace(
            options, ns, w.trace, tag_artifact(graphml_path));
        if (want_yaml)
          violation_yaml_goto_trace(
            options, ns, w.trace, tag_artifact(yaml_path));
        if (want_testcase)
          generate_testcase(
            "testcase-" + witness_id + ".xml", local_eq, *solver_ptr);
        if (want_html)
          generate_html_report(witness_id, ns, w.trace, options);
        if (want_json)
          generate_json_report(witness_id, ns, w.trace);
        if (want_pytest)
          pytest_gen.collect(local_eq, *solver_ptr);
        if (want_ctest)
          ctest_gen.collect(local_eq, *solver_ptr, ns);
        if (want_foundry)
        {
          // ---- A PATH PROVEN REVERT-ONLY MUST NOT BECOME A TEST ----
          //
          // goto_coverage.h states the rule for the obstacle machinery and it
          // applies verbatim here: "a marked path must not be turned into a
          // test. Marking without excluding would be worthless." Recording
          // `arith_revert_only` in the report and then handing the same
          // counterexample to the emitter would reproduce, one mechanism later,
          // exactly the defect a6ea07f2e9 fixed -- where the marking ran on
          // every path, the readers existed, and their gate (`status == U`)
          // excluded the only dangerous case (`status == F`).
          //
          // REFUSED HERE, NOT IN foundry.cpp BESIDE THE OBSTACLE REFUSAL. The
          // reason is that a REFUSAL needs no per-call state: the decision is
          // "do not hand this counterexample over", and this is the only call
          // site that feeds the coverage-mode emitter, so the test sits where
          // the proof was established a few dozen lines above.
          //
          // ⚠ AN EARLIER VERSION OF THIS COMMENT ALSO CLAIMED that putting it
          // in the emitter "would need the fact carried there through a second
          // channel". THAT WAS FALSE and is corrected rather than deleted,
          // because it would have misdirected the next reader.
          // `arith_revert_only_paths` is a static keyed by the SAME
          // (comment, location) pair as `normal_exit_paths` and
          // `named_obstacle_paths`, which foundry.cpp already reads at
          // reconstruct()'s segment site and at its refuted-claim site. There
          // is no second channel; it is the first one.
          //
          // ⚠ AND THERE IS A REAL HAZARD FOR WHOEVER DOES WIRE THE EMITTER,
          // which the obstacle precedent never had to face. Those two sets are
          // filled at INSTRUMENTATION time, single-threaded, so foundry.cpp's
          // unlocked reads of them are safe. THIS one is filled HERE, inside
          // the per-claim job loop, and `--parallel-solving` runs one thread
          // per job; reconstruct() runs unlocked on a job thread and scans
          // every guard-true assert, i.e. OTHER claims' keys, which other jobs
          // may be inserting concurrently. Any read of this set from
          // foundry.cpp must take `claim_outcome_mutex` -- the mutex the write
          // above and the read below both use.
          //
          // COVERAGE IS UNAFFECTED. The path IS witnessed and stays F -- it is
          // reachable on chain, through a Panic revert. What is refused is
          // rendering it as a bare call that ASSERTS a normal exit, which is
          // red on the unmodified contract. Rendering it correctly needs
          // vm.expectRevert(stdError.arithmeticError / divisionError) and is
          // its own change.
          // ---- AND THE REFUSAL IS DEFEATED UNLESS THIS COMES FIRST ----
          //
          // MEASURED on notes/coverage/poc/D16_OnlyByOverflow.sol, which exists
          // to exercise the proof arm. With --overflow-check on, goto_check's
          // OWN overflow claim is another job in this very loop: it is refuted
          // (an overflow IS possible), its witness loop runs, and it calls
          // foundry().collect() like any other claim. The case it reconstructs
          // is the SAME call as the path claim's, so:
          //
          //   without the flag  the two cases share a fingerprint, dedup keeps
          //                     one, and the PATH claim supplies its name --
          //                     the run reports `3 of 3 case(s) name the
          //                     obligation`;
          //   with the flag     the path claim is refused, the arithmetic
          //                     claim's copy survives, and the identical
          //                     wrapping call is emitted anyway as
          //                     `// claim: not recorded` -- the run reports
          //                     `2 of 3`.
          //
          // So refusing the path claim alone moved the defect rather than
          // removing it, and made the artefact WORSE: same red test, now with
          // no provenance back to the report. A claim that is not an
          // instrumented path goal has no business producing a
          // complete-path-coverage test case, and under path coverage that is
          // exactly what these are.
          //
          // Narrowed to path coverage on purpose. Branch coverage's goals carry
          // the same property string, so this cannot change their output; and
          // no path-coverage run passed an arithmetic check before this flag
          // existed (INVOCATION_DECISIONS row 6 says not to), so there is no
          // prior behaviour here to preserve.
          const bool non_path_claim =
            is_path_cov && claim.claim_property != "instrumented assertion";
          if (non_path_claim)
            log_debug(
              "coverage",
              "not collecting a Foundry case for the non-instrumented claim "
              "'{}' ({}): under path coverage only an enumerated path goal may "
              "produce a test",
              claim.claim_msg,
              claim.claim_property);
          bool revert_only = false;
          {
            std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
            revert_only = goto_coveraget::arith_revert_only_paths.count(
                            {claim.claim_msg, claim.claim_loc}) > 0;
          }
          if (non_path_claim)
          {
            // counted separately from the revert-only refusal: they are
            // different facts and a shared counter would make either
            // unreadable.
          }
          else if (revert_only)
          {
            goto_coveraget::arith_revert_only_suppressed.fetch_add(
              1, std::memory_order_relaxed);
            log_warning(
              "--path-cov-arith-resolve: REFUSING to emit a Foundry case for "
              "'{}'. The re-solve PROVED this path is reachable only by "
              "violating a checked arithmetic operation, so on chain it is "
              "reached through a Panic revert -- a bare call asserting a "
              "normal exit would be RED on the unmodified contract. The path "
              "remains witnessed (F) and counted as covered; only the "
              "rendering is refused",
              prettify_solidity_expr(claim.claim_msg));
          }
          else
            // Collect into the external strategy-level generator when threaded
            // (--k-induction), else the owned member (plain BMC).
            foundry().collect(local_eq, *solver_ptr, ns);
        }

        // Solidity complete-path coverage: harvest this path's CE payload
        // (concrete inputs + post-state) NOW, while this witness's solver
        // model is live — after the next dec_solve() it is gone.
        //
        // EVERY witness, not just the first. The old guard was
        // `is_path_cov && witnesses.empty()`, documented as "one CE per
        // complete path is what the report contracts for" — but the contract
        // was the limitation, not a reason for it: with --all-witnesses the
        // model for witnesses 2..N is built, printed and handed to the Foundry
        // generator, and only this harvest threw it away. Default behaviour is
        // unchanged, because without --all-witnesses the loop breaks after one.
        if (is_path_cov)
        {
          // FAULT INJECTION into the window between the two records of one
          // refutation. `claim_outcome[sig] = 'F'` is already written at this
          // point; `reached_claims.emplace(sig)` is not, and will not be if
          // this throws. That is the state P16_Mapping reached by running out
          // of memory here, and the only way a regression can reach it.
          if (fault_mid_witness && decided_claims.load() >= fault_mid_witness)
          {
            log_error(
              "--path-cov-fault-mid-witness {}: injecting std::bad_alloc "
              "INSIDE the counterexample harvest, after {} decided claim(s) "
              "(fault injection; this is not a real allocation failure)",
              fault_mid_witness,
              decided_claims.load());
            throw std::bad_alloc();
          }
          goto_coveraget::path_ce_t ce;
          ce.sliced = !options.get_bool_option("no-slice");
          // Same condition the dispatch uses to set protect_ce_symbols: with
          // the JSON requested, the payload symbols are exempt from slicing, so
          // `sliced` alone must not be read as "the payload was cut down".
          ce.payload_symbols_protected =
            options.get_bool_option("cov-report-json");
          ce.compact_trace = !keep_hidden_steps;
          ce.revert_pre_rollback = goto_coveraget::revert_paths.count(
                                     {claim.claim_msg, claim.claim_loc}) > 0;
          // Ordered so the emitted post-state is deterministic across runs.
          std::map<std::string, std::string> last_state;
          // The EVM environment, LAST write wins -- see the `is_env` branch
          // below for why first-wins was wrong and what it cost. Separate from
          // `input_seen` on purpose: sharing one set lets an environment name
          // and a parameter name suppress each other.
          std::map<std::string, std::string> last_env;
          std::set<std::string> input_seen;
          // Scope prefix this path's function parameters carry. A Solidity
          // parameter symbol is `sol:@C@<contract>@F@<method>@<param>` (see
          // foundry.h's parse_param_symbol), i.e. the method appears WITHOUT
          // the `#N` disambiguator that the goto function id carries. So take
          // claim_msg's function id ("<function-id>:path:<enc>") and drop the
          // `#N` before using it as a prefix, or nothing would ever match.
          std::string fn_scope;
          // Contract scope prefix (`sol:@C@<Contract>@`). Mappings and dynamic
          // arrays do NOT live inside the contract object: the frontend lowers
          // them to CONTRACT-LEVEL global stores, whose ids carry this prefix
          // WITHOUT the `@F@` function marker. That is what separates a state
          // store from a function-scoped local.
          std::string contract_scope;
          // The goto function id, `#N` INTACT — that is the key
          // path_entry_ghost is stored under, so it must not be truncated the
          // way fn_scope is.
          std::string fn_id_full;
          // ---- WHERE A TUPLE RETURN LIVES, AND WHY IT NEEDS ITS OWN KEY ----
          //
          // MEASURED on notes/coverage/poc/P27_TupleReturn.sol: a Solidity
          // function returning `(uint256, uint256)` emits NO `RETURN`
          // instruction at all. The frontend lowers it to per-member writes
          //
          //     ASSIGN tuple_instance$42.mem0 = 11;
          //     ASSIGN tuple_instance$42.mem1 = 12;
          //
          // and falls straight to END_FUNCTION. So the return-value GHOST,
          // which hooks the RETURN instruction, cannot see a tuple by
          // construction -- and the values were instead landing in the
          // contract-scope-store branch below and being published as
          // `state_written_value_unavailable`, i.e. as a state variable named
          // `tuple_instance$42.mem0` that the path supposedly wrote. That is a
          // wrong LABEL on a right value, which is worse than a missing one.
          //
          // The instance is named with the FUNCTION'S OWN AST NODE ID, so it
          // can be tied to this path's unit exactly rather than by matching the
          // string `tuple_instance` anywhere: `two_scalars#42` owns
          // `tuple_instance$42` and `mixed_width#65` owns `tuple_instance$65`
          // (both verified in the same dump). Without that tie an internal
          // callee inlined into this unit would donate its own tuple.
          std::string tuple_want;
          {
            const auto p = claim.claim_msg.rfind(":path:");
            if (p != std::string::npos)
            {
              std::string fn_id = claim.claim_msg.substr(0, p);
              fn_id_full = fn_id;
              const auto fpos = fn_id.find("@F@");
              if (fpos != std::string::npos)
                contract_scope = fn_id.substr(0, fpos + 1);
              const auto hash =
                fn_id.find('#', fpos == std::string::npos ? 0 : fpos + 3);
              if (hash != std::string::npos && !contract_scope.empty())
                tuple_want =
                  contract_scope + "tuple_instance$" + fn_id.substr(hash + 1);
              if (hash != std::string::npos)
                fn_id.erase(hash);
              fn_scope = fn_id + "@";
            }
          }
          // Entry detection, using the SAME algorithm ESBMC already uses for
          // `--show-funccall-trace` (goto_trace.cpp): every trace step carries
          // the call stack at that point, innermost-first. Diff each step's
          // stack against the previous one by common SUFFIX (the outermost end)
          // and whatever remains at the front was pushed by this step. A new
          // invocation of this path's function begins exactly when one of those
          // newly-pushed frames is it.
          //
          // Two earlier attempts were wrong and are recorded so they are not
          // retried. (1) The instrumentation's `tr = 1` is folded into the
          // following `tr = tr*2 + guard`, so it never reaches the trace.
          // (2) Comparing the location's BARE function name cannot tell one
          // contract's `f` from another's, and cannot see a recursive or
          // re-entrant call into the same function at all — it looks like
          // execution never left. Stack frames carry the full goto function id,
          // so both cases are exact.
          //
          // Steps with an EMPTY stack are skipped rather than treated as a
          // reset: goto_trace.cpp documents that an empty stack means "captured
          // outside any function context", not "no frames on the stack", and
          // treating it as a reset re-emits the whole chain spuriously.
          // Recursion makes "the entry state" ambiguous: with `f` nested three
          // deep there are three different entry states, and the assert that
          // the solver refuted belongs to exactly ONE of them. Measured on a
          // path reachable only inside a recursive call, snapshotting on the
          // most recent push reported the INNERMOST invocation's state, which
          // was neither the outer nor the owning one.
          //
          // So key each snapshot by how many frames of this function were on
          // the stack when it was taken, and at the end pick the depth the
          // violated assert itself is at.
          std::vector<stack_framet> prev_stack;
          std::map<size_t, std::map<std::string, std::string>> entry_by_depth;
          size_t prev_target_depth = 0;
          size_t assert_target_depth = 0;
          bool assert_depth_known = false;
          auto count_target_frames =
            [&fn_id_full](const std::vector<stack_framet> &s) {
              size_t n = 0;
              for (const auto &fr : s)
                if (fr.function.as_string() == fn_id_full)
                  ++n;
              return n;
            };
          std::map<std::string, std::string> entry_snapshot;
          // Base symbol id of a possibly-indexed lvalue (`m[k]` -> `m`).
          auto base_sym_id = [](expr2tc e) -> std::string {
            for (;;)
            {
              if (is_index2t(e))
                e = to_index2t(e).source_value;
              else if (is_member2t(e))
                e = to_member2t(e).source_value;
              else
                break;
            }
            return is_symbol2t(e) ? to_symbol2t(e).thename.as_string()
                                  : std::string();
          };
          std::set<std::string> unrendered_seen;
          // The unit's own return value and the runtime record that a RETURN
          // actually executed. Accumulated locally and combined after the walk,
          // for the same reason the environment is: the pair that belongs in
          // the payload is the one in force when the walk stopped at this
          // path's own assert, and a half-updated pair published mid-walk would
          // pair one path's flag with another's value.
          std::string ret_value;
          bool ret_flag = false;
          // A TUPLE return, member index -> value. Keyed on the INDEX rather
          // than the name so `mem10` cannot sort between `mem1` and `mem2`,
          // which a lexicographic map would do silently and which would publish
          // the members of the tuple in the wrong order -- a wrong value
          // wearing the right shape.
          std::map<unsigned long, std::string> ret_members;
          for (const auto &st : w.trace.steps)
          {
            // Stop at THIS path's own violated assert. The harness runs
            // --solidity-max-tx transactions (default 2) and the trace spans
            // all of them, so scanning past this point would let a LATER
            // transaction's writes overwrite the post-state of the invocation
            // that this path actually describes.
            if (st.is_assert() && st.comment == claim.claim_msg)
            {
              ce.scoped_to_claim = true;
              // Which invocation does the refuted assert belong to? Its own
              // stack says so.
              if (!st.stack_trace.empty())
              {
                assert_target_depth = count_target_frames(st.stack_trace);
                assert_depth_known = assert_target_depth > 0;
              }
              break;
            }
            // Entry detection runs on EVERY step, not only assignments: the
            // frame is pushed by the call instruction itself.
            if (!st.stack_trace.empty())
            {
              const auto &cur = st.stack_trace;
              size_t common = 0;
              while (common < prev_stack.size() && common < cur.size() &&
                     prev_stack[prev_stack.size() - 1 - common] ==
                       cur[cur.size() - 1 - common])
                ++common;
              bool pushed_target = false;
              for (size_t k = cur.size() - common; k-- > 0;)
                if (cur[k].function.as_string() == fn_id_full)
                {
                  pushed_target = true;
                  break;
                }
              const size_t d = count_target_frames(cur);
              // Record only on an INCREASE, and only for the depth just
              // reached: whatever state has accumulated is what that
              // particular invocation started from.
              if (pushed_target && d > prev_target_depth)
                entry_by_depth[d] = last_state;
              prev_target_depth = d;
              prev_stack = cur;
            }
            if (!st.is_assignment())
              continue;
            if (is_nil_expr(st.lhs) || is_nil_expr(st.value))
              continue;
            // Diagnostic for the classification below: `--verbosity
            // coverage:9` prints every assignment the harvest sees, with the
            // things the classification keys on. Without it, an empty
            // `inputs`/`entry_storage` is indistinguishable from a rule that
            // silently never matches — which is exactly how the two bugs above
            // (location-based extcall matching, `tr = 1` as entry marker) were
            // found.
            {
              const expr2tc dnd = symex_slicet::get_nondet_symbol(st.rhs);
              log_debug(
                "coverage",
                "ce step: sym='{}' name='{}' fn='{}' nondet='{}'",
                is_symbol2t(st.lhs) ? to_symbol2t(st.lhs).thename.as_string()
                                    : std::string("<not-a-symbol>"),
                from_expr(ns, "", st.lhs),
                st.pc->location.get_function().as_string(),
                (!is_nil_expr(dnd) && is_symbol2t(dnd))
                  ? to_symbol2t(dnd).thename.as_string()
                  : std::string("-"));
              std::string frames;
              for (const auto &fr : st.stack_trace)
                frames +=
                  (frames.empty() ? "" : " < ") + fr.function.as_string();
              log_debug(
                "coverage",
                "   stack[{}]: {}",
                st.stack_trace.size(),
                frames.empty() ? std::string("<empty>") : frames);
            }
            const std::string name = from_expr(ns, "", st.lhs);
            // NOT extcall_returns. The `_ESBMC_Nondet_Extcall_<C>` symbols are
            // the RE-ENTRY model's own choice bits (which public method the
            // outside world calls back into), not the value the external call
            // returned to the contract. Measured: on the two paths of
            // `if (ok)`, which by construction disagree about `ok`, these
            // symbols carry the IDENTICAL tuple — so they cannot be `ok`.
            // Publishing them under `extcall_returns` would be a wrong answer
            // wearing the right label, so they stay counted as plumbing.
            // See ce_extraction.extcall_returns_unavailable_reason below.
            if (
              is_symbol2t(st.lhs) &&
              to_symbol2t(st.lhs).thename.as_string().find(
                "_ESBMC_Nondet_Extcall") != std::string::npos)
            {
              ++ce.dropped_internal;
              continue;
            }
            // ---- THE UNIT'S OWN RETURN VALUE ----
            //
            // Materialised by the instrumenter at the RETURN site
            // (goto_coverage.cpp, "MATERIALISE THE UNIT'S OWN RETURN VALUE"),
            // because it exists nowhere else: the dispatcher calls a unit with
            // no lvalue and the RETURN carries an expression, so before that
            // change no trace step held the value at all -- measured, 208
            // classified assignments on P19_ReturnShapes.tern_lit and not one
            // of them was the return.
            //
            // Taken HERE, ahead of the classification below, for the same
            // reason the extcall symbols are: that classification has exactly
            // three outcomes -- parameter -> inputs, environment -> env,
            // otherwise dropped -- and a ghost is none of the first two, so it
            // would be filed as harness plumbing.
            //
            // A non-constant model value is left UNKNOWN rather than rendered:
            // `final_state` already shipped an unevaluated expression string
            // once ("0xFFFF... / 0") and a consumer parsing this as an integer
            // would refuse it, or worse, print it into a test.
            // The FLAG is checked first and is the only thing that authorises
            // publishing a value. MEASURED with the value alone: tern_lit's
            // revert path (enc=2, which never reaches a RETURN) published the
            // entry initialisation as `return_value: "0"`. 0 is also a real
            // return value, so nothing about the value itself can separate the
            // two cases -- only a runtime record of "a RETURN executed" can.
            if (
              is_symbol2t(st.lhs) &&
              to_symbol2t(st.lhs).thename.as_string().find(
                "__ESBMC_path_retset$") != std::string::npos)
            {
              if (is_constant_expr(st.value))
                ret_flag = !is_false(st.value);
              continue;
            }
            if (
              is_symbol2t(st.lhs) &&
              to_symbol2t(st.lhs).thename.as_string().find(
                "__ESBMC_path_ret$") != std::string::npos)
            {
              // Last write before this path's own assert wins, matching how the
              // environment is harvested. A non-constant model value is left
              // alone rather than rendered: `final_state` already shipped an
              // unevaluated expression string once ("0xFFFF... / 0").
              if (is_constant_expr(st.value))
                ret_value = from_expr(ns, "", st.value);
              continue;
            }
            // Mapping / dynamic-array state: these do NOT live in the contract
            // object, so the `this->` test below never sees them and they would
            // be discarded as harness noise — leaving a reader to infer the
            // mapping was untouched. Recognise them by their contract-level id
            // and record them as state; when the model value is not a constant
            // (an infinite-array store), record the NAME so the write is still
            // visible rather than silently lost.
            const std::string bid = base_sym_id(st.lhs);
            // A member of THIS unit's tuple instance is the unit's own return
            // value, and it must be claimed BEFORE the contract-scope branch
            // below -- that branch's test (`sol:@C@...` with no `@F@`) is true
            // of the tuple instance, which is how these values were being
            // published as unrenderable contract state.
            //
            // No runtime flag is needed here, unlike the scalar case. The
            // members are written only on a path that actually returns (the
            // revert arm of P27's `two_scalars` jumps straight past them), so
            // the presence of the write IS the runtime evidence the flag
            // supplies for a RETURN.
            if (
              !tuple_want.empty() && bid == tuple_want && is_member2t(st.lhs) &&
              is_constant_expr(st.value))
            {
              const irep_idt &memid = to_member2t(st.lhs).member;
              const std::string mem = memid.as_string();
              // PROJECT. A tuple member write is lowered as a WHOLE-OBJECT
              // update, exactly like the `this->x = v` case a few lines below,
              // so the model hands back the ENTIRE tuple struct. MEASURED
              // without this projection on P27_TupleReturn.two_scalars:
              //     mem0 -> "{ .mem0=11, .mem1=0 }"
              //     mem1 -> "{ .mem0=11, .mem1=12 }"
              // i.e. every member reporting the whole aggregate, and the
              // rendered "tuple" being a pair of structs rather than a pair of
              // values. On the unequal-width unit it additionally exposed the
              // struct's `anon_pad$2` padding member.
              expr2tc mv = st.value;
              if (is_constant_struct2t(mv))
              {
                const struct_type2t &sty = to_struct_type(mv->type);
                for (size_t i = 0; i < sty.member_names.size(); ++i)
                  if (sty.member_names[i] == memid)
                  {
                    mv = to_constant_struct2t(mv).datatype_members[i];
                    break;
                  }
              }
              if (
                mem.rfind("mem", 0) == 0 && is_constant_expr(mv) &&
                !is_constant_struct2t(mv) && !is_constant_array2t(mv))
              {
                char *endp = nullptr;
                const unsigned long k = strtoul(mem.c_str() + 3, &endp, 10);
                if (endp != nullptr && *endp == '\0')
                {
                  ret_members[k] = from_expr(ns, "", mv);
                  continue;
                }
              }
              // A member whose name is not `mem<N>` is NOT silently folded in:
              // falling through leaves it to the branch below, which reports it
              // as written-and-unrenderable. Publishing it as part of the tuple
              // would put an unknown component in a value a test asserts on.
            }
            if (
              !contract_scope.empty() && bid.rfind(contract_scope, 0) == 0 &&
              bid.find("@F@") == std::string::npos)
            {
              // ESBMC's own contract-scope plumbing (address-binding tables,
              // call temporaries) is not user state.
              if (
                name.rfind("$", 0) == 0 ||
                name.find("_bind_cname") != std::string::npos ||
                name.find("return_value$") != std::string::npos)
              {
                ++ce.dropped_internal;
                continue;
              }
              // A mapping/array element write (`m[k] = v`) yields a SCALAR
              // model value — that is the useful post-state. The companion
              // whole-store assignment (`m = with(m,k,v)`) yields the entire
              // infinite array, which is the same information in a form no
              // consumer can use and which would dwarf the report; name it as
              // written-but-unrendered instead of dumping thousands of zeros.
              // Only an ELEMENT write (`m[k] = v`) carries a usable post-state
              // value. A bare whole-store assignment is the `with(...)` bulk
              // form of the same update; recording it would print the entire
              // infinite array — and its zero-initialised form reads as "the
              // mapping is empty", contradicting the element entry beside it.
              const bool scalar = is_index2t(st.lhs) &&
                                  is_constant_expr(st.value) &&
                                  !is_constant_array2t(st.value) &&
                                  !is_constant_struct2t(st.value);
              if (scalar)
                last_state[name] = from_expr(ns, "", st.value);
              else if (unrendered_seen.insert(name).second)
                ce.state_written_unrendered.push_back(name);
              continue;
            }
            // Never guess: a value that did not come back from the model as a
            // constant is dropped rather than rendered symbolically.
            if (!is_constant_expr(st.value))
              continue;
            // A Solidity state write `this->x = v` is lowered as a whole-object
            // update, so the model hands back the ENTIRE contract struct.
            // Project out the written member, otherwise every state variable
            // would be reported as the same opaque object literal.
            expr2tc val_expr = st.value;
            if (is_member2t(st.lhs) && is_constant_struct2t(val_expr))
            {
              const irep_idt &mem = to_member2t(st.lhs).member;
              const struct_type2t &sty = to_struct_type(val_expr->type);
              for (size_t i = 0; i < sty.member_names.size(); ++i)
                if (sty.member_names[i] == mem)
                {
                  val_expr = to_constant_struct2t(val_expr).datatype_members[i];
                  break;
                }
            }
            // ---- THE GUARD ABOVE TESTS THE WRONG EXPRESSION ----
            //
            // `is_constant_expr(st.value)` is checked BEFORE the projection,
            // and a Solidity state write is lowered as a whole-object update:
            // the model hands back the entire contract struct, which IS a
            // constant_struct, so the test passes. The value actually published
            // is the PROJECTED MEMBER, and that can be anything the simplifier
            // declined to fold.
            //
            // MEASURED on notes/coverage/poc/P18_Unchecked.sol, unit `div`,
            // path 6:
            //
            //     final_state {"r": "0xFFFF...FFFF / 0"}
            //
            // -- an unevaluated division, as a STRING, in a field contracted to
            // hold values. The simplifier explicitly refuses to fold a zero
            // divisor (smt_conv), so `a / 0` never becomes a constant and
            // from_expr renders all of it. `--div-by-zero-check` does NOT
            // change it: that cell is byte-identical, because the check adds a
            // claim and constrains no model.
            //
            // EXECUTION_PLAN §7.1 predicted this one step later -- it expected
            // `type(uint256).max`, bvudiv's total-function value -- and warned
            // the value would flow into R2 assertions. The R1/R2 ladder is
            // built from this exact field, and a consumer either parses it as
            // an integer (refusing the coordinate on failure, safe but lossy)
            // or renders it into a test.
            //
            // Dropped and NAMED, through the channel that already exists for
            // "this path wrote it and the value is not renderable" -- the same
            // one mapping and dynamic-array stores use. Silently omitting it
            // would let a reader infer the variable was UNCHANGED, which is a
            // wrong conclusion rather than a missing one.
            if (!is_constant_expr(val_expr))
            {
              const std::string bare_un =
                name.find("this->") != std::string::npos
                  ? name.substr(name.rfind("->") + 2)
                  : name;
              // THE SAME PLUMBING FILTER ITS TWO NEIGHBOURS USE. Without it
              // this reported `_ESBMC_Object_<C>.$address` and
              // `_ESBMC_bind_cname` as user state the path wrote and could not
              // render -- measured on P18 the moment the drop was added. Those
              // are ESBMC's own address-binding fields; the contract-scope
              // branch above drops them as `dropped_internal` and the `this->`
              // branch below refuses them by the same two tests, so a third
              // spelling of the rule would have been a third thing to keep in
              // step. Reporting them would be noise in a list whose whole value
              // is that every name in it is a real state variable.
              if (
                bare_un.empty() || bare_un[0] == '$' ||
                bare_un.rfind("_ESBMC", 0) == 0 ||
                bare_un.find("_bind_cname") != std::string::npos ||
                bare_un.find("return_value$") != std::string::npos)
              {
                ++ce.dropped_internal;
                continue;
              }
              if (unrendered_seen.insert(bare_un).second)
                ce.state_written_unrendered.push_back(bare_un);
              continue;
            }
            const std::string val = from_expr(ns, "", val_expr);
            // WHOLE-OBJECT restore: require(cond) / revert("msg") lower to a
            // rollback block that assigns the entry snapshot back as ONE
            // aggregate — `*this = _sol_save_this` — never per field. Tracking
            // only `this->member` writes would miss it and report the
            // pre-rollback value as the post-state, i.e. a WRONG value. Adopt
            // every member of the restored object as the new post-state.
            if (name.rfind("*this", 0) == 0 && is_constant_struct2t(st.value))
            {
              const struct_type2t &sty = to_struct_type(st.value->type);
              const auto &mems =
                to_constant_struct2t(st.value).datatype_members;
              for (size_t i = 0; i < sty.member_names.size() && i < mems.size();
                   ++i)
              {
                const std::string mn = sty.member_names[i].as_string();
                if (
                  mn.empty() || mn[0] == '$' || mn.rfind("_ESBMC", 0) == 0 ||
                  mn.rfind("anon_pad", 0) == 0)
                  continue;
                if (is_constant_expr(mems[i]))
                  last_state[mn] = from_expr(ns, "", mems[i]);
              }
              continue;
            }
            if (name.find("this->") != std::string::npos)
            {
              // Contract state variable: the LAST write on this path is the
              // post-state this path produces. ESBMC's own bookkeeping fields
              // on the contract object ($address/$code/$codehash/binding) are
              // not user state and would only add noise.
              const std::string bare = name.substr(name.rfind("->") + 2);
              if (
                !bare.empty() && bare[0] != '$' && bare.rfind("_ESBMC", 0) != 0)
                last_state[bare] = val;
            }
            else
            {
              // A nondet-sourced assignment is a value the harness CHOSE. But
              // "chosen by the harness" is far broader than "call argument":
              // it also covers the allocator tables, dispatcher choice bits and
              // temporaries. Classify instead of dumping the whole bag, which
              // would bury the one value a consumer needs (and bloat the report
              // with multi-KB internal arrays).
              const expr2tc nd = symex_slicet::get_nondet_symbol(st.rhs);
              if (is_nil_expr(nd))
                continue;
              // A Solidity parameter/local carries the mangled id
              // `sol:@C@<Contract>@F@<fn>#N@<name>`. Requiring the prefix to be
              // THIS path's own function id keeps the call arguments of the
              // unit under test and drops harness/dispatcher locals that merely
              // share the shape (`_new_ts`, `addr`, `tmp`, ...).
              std::string sym_id;
              if (is_symbol2t(st.lhs))
                sym_id = to_symbol2t(st.lhs).thename.as_string();
              const bool in_fn_scope =
                !fn_scope.empty() && sym_id.rfind(fn_scope, 0) == 0;
              // A prefix match says "declared inside this function", which is
              // true of a PARAMETER and of every local alike. Only a parameter
              // is something the caller supplies, so ask the symbol table
              // instead of the name. Before this, an external call's return
              // value (a local) was reported under `inputs`, i.e. as an
              // argument a generated test could pass — it cannot.
              bool is_param = false;
              if (in_fn_scope && !sym_id.empty())
              {
                const symbolt *psym = ns.lookup(irep_idt(sym_id));
                is_param = psym && psym->is_parameter;
              }
              // ⚠ This used to read "external-call returns were already taken
              // above, before the contract-scope-store branch could discard
              // them". No such code exists anywhere -- ce.extcall_returns has a
              // declaration, three readers and ZERO writers -- so the comment
              // asserted a step that had never been written. It cost a session:
              // the empty field was read as "the harvest cannot see the value",
              // when in fact the value arrives here resolved and falls off the
              // end of the classification below for want of a bucket.
              // A comment claiming a step that does not exist is worse than no
              // comment, because it redirects the next reader away from the
              // line that actually drops the value.
              const bool is_env = name.rfind("msg_", 0) == 0 ||
                                  name.rfind("tx_", 0) == 0 ||
                                  name.rfind("block_", 0) == 0;
              if (is_param)
              {
                if (input_seen.insert(name).second)
                  ce.inputs.emplace_back(name, val);
              }
              else if (is_env)
              {
                // ---- LAST WRITE WINS FOR THE ENVIRONMENT, NOT FIRST ----
                //
                // A parameter is assigned once, at the call, so first-wins is
                // right for `inputs`. The EVM environment is not: it is written
                // once at declaration (`solidity_blockchain.c`, before the
                // harness runs) and then RE-SEEDED at the top of every
                // dispatcher iteration by `_sol_per_tx_reseed`. First-wins
                // therefore always published the declaration-time value and
                // silently discarded the one the transaction actually ran
                // under.
                //
                // MEASURED on notes/coverage/poc/D09_ValueGate.sol -- six
                // lines, one unit, no source decision, so the only decision in
                // the run is the synthetic ABI value gate. From the
                // `set:path:2` counterexample, verbatim:
                //
                //     State 11  solidity_blockchain.c:31
                //       msg_value = 0
                //     State 51  solidity_misc.c:171  _sol_per_tx_reseed
                //       msg_value = 0xFFFF...FFFF
                //     State 72  D09_ValueGate.sol:36  set
                //       path_tr$0 = 2
                //     Violated property: ...:path:2
                //
                // `path:2` is the depth-1 revert taken when value is sent to a
                // NONPAYABLE entry, so it requires `msg.value != 0`, and the
                // report published 0 -- the condition under which the path is
                // NOT taken. The sibling `set:path:3` is the control: there the
                // reseed chooses 0 and 0 is correct, so the old code was right
                // half the time by coincidence.
                //
                // THE COST WAS NOT A MISSING FIELD. Both arms of the gate then
                // render as the same call, so ONE emitted Foundry case carries
                // BOTH path ids -- 37 of 161 cases across the PoC set -- and a
                // path no test can reach is counted as covered. The same
                // mechanism reseeds msg_sender (State 50), which is the other 6
                // of the 63 payload-vs-path contradictions
                // `notes/coverage/scripts/ce_consistency.py` reports, so 61 of
                // the 63 are this one line.
                //
                // "LAST" IS BOUNDED BY THE LOOP, NOT BY THE TRACE. This walk
                // already breaks at this path's own violated assert (see the
                // `st.comment == claim.claim_msg` guard above), so the last
                // write kept here is the reseed of the transaction that ENTERED
                // the unit -- not a later transaction's. That is what makes
                // this correct at `--solidity-max-tx 2` and not only at 1.
                // Solidity cannot assign msg.* inside a function, so there is
                // no write between entry and the assert to prefer over the
                // reseed.
                //
                // Its own map, no longer sharing `input_seen` with the
                // parameters: one shared set means an environment name and a
                // parameter name that happen to collide silently suppress each
                // other, and `std::map` additionally makes the published order
                // deterministic across runs rather than trace-order.
                last_env[name] = val;
              }
              else
                ++ce.dropped_internal;
            }
          }
          for (const auto &[n, v] : last_state)
            ce.final_state.emplace_back(n, v);
          // BOTH conditions. The flag says a RETURN executed on this path; the
          // non-empty string says its value came back from the model as a
          // constant. Either one alone would publish something the other
          // contradicts.
          if (ret_flag && !ret_value.empty())
          {
            ce.return_value = ret_value;
            ce.return_value_known = true;
          }
          // A tuple return, rendered in MEMBER ORDER as `(v0, v1, ...)`. A unit
          // returns either a single value or a tuple, never both, so the two
          // arms cannot fight over the field; the `else if` says so rather than
          // leaving a last-writer-wins that would silently prefer one shape.
          else if (!ret_members.empty())
          {
            std::string t = "(";
            bool first = true;
            for (const auto &[k, v] : ret_members)
            {
              t += (first ? "" : ", ") + v;
              first = false;
            }
            ce.return_value = t + ")";
            ce.return_value_known = true;
          }
          // Published after the walk, for the same reason `final_state` is: the
          // value that belongs in the report is the one in force when the unit
          // ran, and that is only known once the walk has stopped at this
          // path's own assert.
          for (const auto &[n, v] : last_env)
            ce.env.emplace_back(n, v);
          // Pick the snapshot for the invocation the refuted assert sits in.
          // Anything else would be a different invocation's entry state, which
          // for a recursive function is a different number — reporting it would
          // be a wrong value, not an approximate one.
          if (assert_depth_known)
          {
            auto ed = entry_by_depth.find(assert_target_depth);
            if (ed != entry_by_depth.end())
            {
              entry_snapshot = ed->second;
              ce.entry_storage_known = true;
            }
          }
          for (const auto &[n, v] : entry_snapshot)
            ce.entry_storage.emplace_back(n, v);
          // Accumulated locally and published once after the loop: publishing
          // per witness would leave the shared maps disagreeing about how many
          // witnesses a claim has for the duration of the enumeration, and the
          // CE journal is written from those maps by a hook on this same claim.
          ce_all.push_back(std::move(ce));
        }

        witnesses.push_back(std::move(w));

        if (!enumerate)
          break;
        if (witnesses.size() >= max_w)
        {
          stop_reason = enumeration_stop_reasont::CapHit;
          break;
        }

        // If this witness has no nondet inputs we can't enumerate further —
        // there's nothing meaningful to block. Mark the reason so the user
        // doesn't read "UNSAT" as "exhaustive".
        if (witnesses.back().nondet_inputs.empty())
        {
          stop_reason = enumeration_stop_reasont::NoInputs;
          break;
        }

        // Open a single SMT context frame the first time we add a blocking
        // clause. Every subsequent blocking clause goes into the same frame;
        // the matching pop_ctx() after the loop drops them all in one shot.
        // This keeps the feature safe under --smt-during-symex, where
        // solver_ptr aliases the shared runtime_solver: blocking clauses
        // asserted while enumerating claim A cannot leak into claim B.
        // (Push must come *after* the first model read — bitwuzla and other
        // backends invalidate the current model on push.)
        if (!ctx_pushed)
        {
          solver_ptr->push_ctx();
          ctx_pushed = true;
        }

        // Block this input tuple and re-solve on the same instance.
        expr2tc block = make_blocking_expr(witnesses.back().nondet_inputs);
        solver_ptr->assert_expr(block);
        enum_result = solver_ptr->dec_solve();
      }

      // dec_solve() can return P_ERROR / P_SMTLIB; in that case the witness
      // set is *not* exhaustive — flag it explicitly.
      if (
        stop_reason == enumeration_stop_reasont::Unsat &&
        enum_result != P_UNSATISFIABLE && enum_result != P_SATISFIABLE)
        stop_reason = enumeration_stop_reasont::Error;

      // Drop every blocking clause we asserted; the next claim's solve
      // sees the solver in its pre-enumeration state.
      if (ctx_pushed)
        solver_ptr->pop_ctx();

      // Publish the payloads. `path_ce` keeps the FIRST witness, unchanged, so
      // every existing consumer (the Foundry emitter, audit_certify_witness,
      // the CE journal, the covered set) reads exactly what it read before;
      // `path_ce_all` carries the whole set. Two maps rather than one changed
      // type, because a changed type is a change to four consumers at once and
      // three of them want one witness.
      if (is_path_cov && !ce_all.empty())
      {
        std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
        goto_coveraget::path_ce[claim_sig] = ce_all.front();
        goto_coveraget::path_ce_all[claim_sig] = std::move(ce_all);
      }

      // Store claim signature (once — multiple witnesses are still one claim)
      if (is_assert_cov)
      {
        std::lock_guard lock(reached_mul_claims_mutex);
        reached_mul_claims.emplace(claim_sig);
      }
      else
      {
        std::lock_guard lock(reached_claims_mutex);
        if (is_goto_cov)
          reached_claims.emplace(claim_sig);
        else
          reached_claims.emplace(claim.claim_key);
        // Item 2e: persist this newly-witnessed branch edge immediately
        // and atomically, so a mid-run kill (the heavy --coverage-whole-
        // unit case dies in-solve before report_coverage) still saves
        // partial progress and bounded re-runs accumulate monotonically.
        // Branch coverage only; key == Item 2's (claim_msg, claim_loc);
        // under the same mutex as reached_claims.
        // Solidity complete-path coverage uses the SAME cross-run covered-set
        // so an escalation round (tx=1 -> 2 -> 3) does not re-instrument a
        // path whose CE is already in hand; its key is the same
        // (claim_msg, claim_loc) pair, where claim_msg is the "fn:path:enc"
        // comment, unique per complete path.
        // Complete-path coverage persists through the CONTENT-ADDRESSED writer
        // instead: its key is the stable id, and `covered_set` is empty for
        // this metric by construction. Incremental (not just at run end) for
        // the same reason as branch coverage — a mid-solve kill on a large
        // contract must not throw away the paths already witnessed.
        // Signal-safe numerator. Bumped under reached_claims_mutex, beside the
        // insertion it counts, so the two cannot drift; read only by the kill
        // handler, which cannot walk reached_claims (a std::unordered_set,
        // possibly mid-rehash when the signal lands).
        if (is_path_cov)
          goto_coveraget::live_F.fetch_add(1, std::memory_order_relaxed);
        if (is_path_cov && !goto_coveraget::path_covered_outpath.empty())
          goto_coveraget::write_path_covered_set_atomic(
            fmt::format(
              "mid-solve after claim {} of {}", decided_now, remaining_claims));
        // THE PAYLOAD, ON DISK, NOW. Unlike the covered set above this needs no
        // opt-in flag, because the run that lost five witnesses to an OOM was
        // not passing one and no collector in this project ever has. Written
        // here rather than at the end of the run for the only reason that
        // matters: the end of the run may not happen.
        if (is_path_cov && !goto_coveraget::path_ce_journal_path.empty())
          goto_coveraget::write_path_ce_journal_atomic(
            fmt::format("after claim {} of {}", decided_now, remaining_claims),
            /*complete=*/false);
        if (
          is_branch_cov && !goto_coveraget::covered_set_outpath.empty() &&
          goto_coveraget::covered_set.emplace(claim.claim_msg, claim.claim_loc)
            .second)
        {
          goto_coveraget::write_covered_set_atomic();
          // "data even on UNKNOWN", covered-set-mode numerator: one
          // more universe edge witnessed+persisted this run. A loaded
          // prior set only raises true coverage, so this running count
          // is a sound lower bound on the covered-set authoritative
          // |all_claims ∩ (covered_set ∪ reached_claims)|.
          goto_coveraget::covered_run.fetch_add(1, std::memory_order_relaxed);
        }
        // "data even on UNKNOWN", default-mode numerator: mirror the
        // canonical bmc.cpp:901 reached_claims.size() (intentionally
        // incl. non-universe entries). Under reached_claims_mutex so
        // the size() read is consistent; grows only here (erases happen
        // exclusively inside report_coverage, which re-syncs the active
        // counter afterwards).
        if (is_branch_cov)
          goto_coveraget::live_reached.store(
            reached_claims.size(), std::memory_order_relaxed);
      }

      // for verbose output of cond coverage.
      // NOT for path coverage: report_coverage_verbose only knows the four
      // legacy metrics and its else-branch is `log_error(...); abort()`, so
      // passing `--verbosity coverage:N` alongside --solidity-path-coverage
      // used to kill the run with a core dump the moment the first path was
      // witnessed. Path coverage already prints per-claim progress through the
      // PASSED/FAILED lines above.
      if (is_vb && !is_path_cov)
        report_coverage_verbose(
          claim,
          claim_sig,
          is_assert_cov,
          is_cond_cov,
          is_branch_cov,
          is_branch_func_cov,
          is_k_path_cov,
          reached_claims,
          reached_mul_claims);
      else if (!is_cov_silent)
      {
        // For a coverage probe the trace is the evidence of reachability —
        // which values drive execution to the goal — so it stays, but framed
        // as a reachability witness rather than a counterexample.
        report_multi_property_trace(
          P_SATISFIABLE, witnesses, stop_reason, claim.claim_msg, is_cov_goal);
      }

      // No claim of a coverage run drives a verdict: the program was never
      // checked against the assertions the instrumentation replaced. A
      // path-coverage query's instrumented claims are its properties.
      if (
        !is_goto_cov || (goto_coveraget::path_cov_query_mode() &&
                         claim.claim_property == "instrumented assertion"))
      {
        std::lock_guard lock(result_mutex);
        final_result = solver_result;
      }

      // Update fail-fast-counter
      fail_fast_cnt++;

      // for kind && incr: remove verified claims
      // whenever we find a property violation, we remove the claim
      if (!is_keep_verified && (bs || fc || is))
      {
        clear_verified_claims_in_ssa(local_eq, claim, is_goto_cov);
        clear_verified_claims_in_goto(claim, is_goto_cov);
      }
    }
    else if (solver_result == P_UNSATISFIABLE)
      // for kind && incr: remove verified claims
      // when we find a property proven correct in
      // either forward condition or inductive step; a claim whose proof is
      // withheld stays for the next base case to solve.
      if (clears_proved_claims)
      {
        clear_verified_claims_in_ssa(local_eq, claim, is_goto_cov);
        clear_verified_claims_in_goto(claim, is_goto_cov);
      }
  };

  // ---- THE REPORT MUST OUTLIVE THE JOB LOOP ----
  //
  // report_coverage sits AFTER this loop and INSIDE run_thread's try
  // (bmc.cpp:2405), and the only verification-phase catch is at :2559. So an
  // allocation failure in any job unwound straight past the report: a CAUGHT
  // OOM cost the ENTIRE report, not part of it. Measured on aqua at 8 g -- 938
  // decided claims and 5 of 15 witnesses, discarded at 51.5% completion.
  //
  // The fix is to make the tail run on the exception path too, then rethrow so
  // nothing downstream sees a different control flow than before. It is
  // deliberately NOT a per-job catch that swallows the failure and continues:
  // when memory is genuinely exhausted the next job throws too, and a loop that
  // absorbs N allocation failures reports N claims as "solver-unknown" while
  // spending the rest of the budget failing. Stopping and keeping the work is
  // the honest behaviour.
  // ---- THE RESCUE MUST NOT NEED MEMORY IT NO LONGER HAS ----
  //
  // MEASURED, and it is the whole reason this exists: with the catch below in
  // place but no cushion, the aqua whole-contract run at 8 g DID reach the
  // rescue, printed "Writing a PARTIAL report with the 938 of 1822 claim(s)
  // decided so far", got as far as the [Coverage] block -- and then threw a
  // SECOND std::bad_alloc while building the JSON, because building a report
  // for 2846 claims needs tens of megabytes and the process had just failed to
  // get any. `cov-report.json` was still not written. A rescue that allocates
  // at the moment allocation is impossible is not a rescue.
  //
  // So a block is reserved BEFORE the solve and released as the first act of
  // the rescue, returning it to the allocator's free list where the report can
  // reuse it without growing the data segment (which is what --memlimit caps:
  // RLIMIT_DATA, esbmc_parseoptions.cpp:691-708).
  //
  // Deliberately NOT touched: untouched pages cost address space, which is the
  // resource under pressure, and no RSS. Sized against the measurement above --
  // aqua's 2846-claim report is ~1.6 MB of text and tens of MB of DOM, so 128
  // MiB is comfortable at 1.6% of an 8 g budget. Only allocated for path
  // coverage, which is the mode that dies this way.
  std::unique_ptr<char[]> oom_cushion;
  if (is_path_cov)
  {
    try
    {
      oom_cushion.reset(new char[128u * 1024u * 1024u]);
    }
    catch (const std::bad_alloc &)
    {
      // Already too tight to reserve. Say so rather than proceed silently: it
      // predicts that the partial report will not fit either.
      log_warning(
        "could not reserve the 128 MiB rescue cushion; if this run dies of an "
        "allocation failure the PARTIAL report may not fit in memory either");
    }
  }

  auto emit_partial = [&](const std::string &reason) {
    // FIRST, before anything that allocates.
    oom_cushion.reset();
    log_error(
      "the per-claim solve loop did not finish ({}). Writing a PARTIAL report "
      "with the {} of {} claim(s) decided so far, rather than discarding them. "
      "It is marked partial in the JSON (`partial: true`, and the same under "
      "`summary`) and on stdout; it must NOT be read as a measurement of this "
      "program",
      reason,
      decided_claims.load(),
      remaining_claims);
    // The rescue gets its own handler so that a failure INSIDE it cannot
    // replace the original reason with its own. Without this, a second
    // bad_alloc thrown while building the JSON propagates out of the catch
    // clause in place of the `throw;` below, and the log then blames the
    // report writer for a run that died in the solver -- the wrong cause,
    // reported confidently.
    try
    {
      if (
        bs && !fc && !is && !options.get_bool_option("k-induction") &&
        !options.get_bool_option("incremental-bmc"))
        report_coverage(
          options,
          reached_claims,
          reached_mul_claims,
          pytest_gen,
          ctest_gen,
          foundry_gen,
          reason);
    }
    catch (const std::bad_alloc &)
    {
      log_error(
        "the PARTIAL report itself could not be built: there was not enough "
        "memory left to serialise it, even after releasing the rescue cushion. "
        "{} of {} claim(s) had been decided. The counterexample payloads of "
        "every path witnessed so far are still on disk in cov-ce-journal.json, "
        "which is written per witness and needs no allocation at this point; "
        "that file, NOT this run's absent cov-report.json, is what survived",
        decided_claims.load(),
        remaining_claims);
    }
    catch (...)
    {
      log_error(
        "the PARTIAL report itself failed to be written. {} of {} claim(s) had "
        "been decided; the witnesses are in cov-ce-journal.json",
        decided_claims.load(),
        remaining_claims);
    }
  };
  try
  {
    // PARALLEL
    if (options.get_bool_option("parallel-solving"))
    {
      /* NOTE: I would love to use std::for_each here, but it is not giving
       * the result I would expect. My guess is either compiler version
       * or some magic flag that we are not using.
       *
       * Nevertheless, we can achieve the same results by just creating
       * threads.
       */

      // TODO: Running everything in parallel might be a bad idea.
      //       Should we also add a thread pool?
      std::vector<std::thread> parallel_jobs;
      for (const auto &i : jobs)
        parallel_jobs.push_back(
          std::thread(
            [&](const size_t &n) {
              run_job_guarded(
                job_function, n, result_mutex, final_result, report_incomplete);
            },
            i));

      // Main driver
      for (auto &t : parallel_jobs)
      {
        t.join();
      }
      // We could remove joined jobs from the parallel_jobs vector.
      // However, its probably not worth for small vectors.
    }
    // SEQUENTIAL
    else
      std::for_each(std::begin(jobs), std::end(jobs), job_function);
  }
  // Typed arms, not a bare catch(...), because the REASON is the point: a
  // reader of a partial report has to be able to tell an out-of-memory kill
  // from a frontend error from an internal invariant, and "unknown exception"
  // is the answer that sends them back to the log they no longer have.
  catch (const std::bad_alloc &)
  {
    emit_partial(
      "std::bad_alloc — the process ran out of memory during the "
      "per-claim solve");
    throw;
  }
  catch (const std::string &e)
  {
    emit_partial("error: " + e);
    throw;
  }
  catch (const char *e)
  {
    emit_partial(std::string("error: ") + e);
    throw;
  }
  catch (const std::exception &e)
  {
    emit_partial(std::string("exception: ") + e.what());
    throw;
  }
  catch (...)
  {
    // Reached only by a throw of a type nothing here knows about. Kept rather
    // than dropped: the alternative is that the one exception nobody
    // anticipated is also the one that costs the whole report.
    emit_partial(
      "an exception of unrecognised type escaped the per-claim "
      "solve loop");
    throw;
  }

  // A warm run solves nothing, so without this the report shows no solver
  // activity at all and gives no sign of where the verdicts came from.
  log_proof_cache_summary(proof_cache.get());

  // One claim reaches the job loop once per physical copy of its unit; the
  // copies of a claim already witnessed are not solved again.
  if (is_path_cov && skipped_witnessed_copies > 0)
    log_status(
      "--solidity-path-coverage: {} copy(ies) of already-witnessed path "
      "claim(s) were not solved again",
      skipped_witnessed_copies.load());

  // For coverage with fixed bound unwinding
  if (
    bs && !fc && !is && !options.get_bool_option("k-induction") &&
    !options.get_bool_option("incremental-bmc"))
    report_coverage(
      options,
      reached_claims,
      reached_mul_claims,
      pytest_gen,
      ctest_gen,
      foundry_gen);

  return final_result;
}

void bmct::seed_property_verdicts(const symex_target_equationt &eq) const
{
  // A coverage or dead-code run fills the same table with reachability probes
  // rather than properties, and renders them in its own vocabulary; seeding
  // would invent goals it never instrumented.
  if (
    options.get_bool_option("coverage-measurement") ||
    options.get_bool_option("dead-code-check"))
    return;

  // A k-step strategy's forward condition asks whether the loop is exhausted,
  // and asks it with an unwinding assertion. That is the strategy's own
  // device, not a property of the program, and belongs in the run's one table
  // no more than "Checking forward condition" does. Only that phase enables
  // unwinding assertions -- every other one sets --no-unwinding-assertions --
  // so the filter needs no further scoping.
  const bool skip_unwinding_assertions =
    strategy_owns_property_table(options) &&
    options.get_bool_option("forward-condition");

  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assert())
      continue;

    if (
      skip_unwinding_assertions &&
      id2string(step.comment).find("unwinding assertion") != std::string::npos)
      continue;

    const std::string description = id2string(step.comment);
    goto_functionst::property_verdicts.record(
      property_key(*step.source.pc, description),
      property_verdictt::NotChecked,
      property_location(*step.source.pc, description));
  }
}

bool bmct::reports_final_verdict(smt_resultt res) const
{
  const bool fc = options.get_bool_option("forward-condition");
  const bool is = options.get_bool_option("inductive-step");

  // A k-step run prints one table. A base case deepens rather than concludes
  // -- the loop goes on to k+1 whatever it finds -- and the diagnostic pass
  // runs after the strategy has already given up, so the driver reports both.
  if (
    is_bounded_round(options) ||
    (strategy_owns_property_table(options) &&
     options.get_bool_option("diagnose-unknown-properties")))
    return false;

  if (res == P_SATISFIABLE)
    return !is && !fc;

  if (res != P_UNSATISFIABLE)
    return false;

  if (is && options.get_bool_option("termination"))
    return true;

  // An intermediate round of an iterative strategy deepens rather than
  // concludes; reporting here would print one table per k.
  return !options.get_bool_option("base-case") &&
         !options.get_bool_option("suppress-bounded-success");
}

bool bmct::all_properties_proved(smt_resultt res) const
{
  // Stricter than report_result() on report_incomplete alone: a run cut short
  // by --multi-fail-fast or --multi-property-interleavings still reports
  // SUCCESSFUL, having found no violation, but claims it never solved are not
  // thereby proved, and promoting them would invent per-property results the
  // run's own "report is partial" note contradicts.
  if (
    res != P_UNSATISFIABLE || report_incomplete || vacuity_detected ||
    ltl_uninstrumented)
    return false;

  // Modes whose SAT/UNSAT is not a statement about the program's properties,
  // and rounds already known not to prove them.
  static const char *const disqualifying[] = {
    "k-induction-parallel",
    "diagnose-unknown-properties",
    "coverage-measurement",
    "dead-code-check",
    "suppress-bounded-success"};
  for (const char *option : disqualifying)
    if (options.get_bool_option(option))
      return false;

  // A violation found at an earlier k settles that claim and no other. Under
  // --multi-property each claim is solved on its own at every base case, so
  // the ones this phase proves are proved; without it the run encodes one
  // formula, and a violation leaves the rest of it undecided.
  if (
    options.get_bool_option("kind-violation-found") &&
    !strategy_owns_property_table(options))
    return false;

  const bool is = options.get_bool_option("inductive-step");
  if (
    is && (options.get_bool_option("termination") ||
           options.get_bool_option("disable-inductive-step")))
    return false;

  // A base case alone is bounded: it refutes a bug up to k, it does not prove
  // the property. A single --multi-property run has no larger k to come, so
  // its per-claim UNSATs are proofs; under a k-step strategy they are not,
  // and the forward condition or the inductive step is what settles them.
  return !options.get_bool_option("base-case") ||
         reports_multi_property_verdict(options);
}

/// A coverage run records probes in the same table, but a probe is a
/// reachability question, not a property: SAT means the location is reachable,
/// so it must not be labelled a violation (issue #6387). Rendered separately,
/// and deliberately unchanged, so a coverage report keeps its own vocabulary.
void bmct::report_coverage_goal_verdicts(
  const std::map<std::string, property_resultt> &verdicts) const
{
  const bool is_color = options.get_bool_option("color");
  const std::string GREEN = is_color ? "\033[32m" : "";
  const std::string YELLOW = is_color ? "\033[33m" : "";
  const std::string RESET = is_color ? "\033[0m" : "";

  size_t unreached = 0, reached = 0, undecided = 0;
  for (const auto &[property, result] : verdicts)
  {
    const char *label = nullptr;
    const std::string *color = nullptr;
    switch (result.verdict)
    {
    case property_verdictt::Passed:
    case property_verdictt::NotChecked:
      ++unreached;
      label = "- UNREACHED";
      color = &YELLOW;
      break;
    case property_verdictt::Unknown:
      ++undecided;
      label = "? UNDECIDED";
      color = &YELLOW;
      break;
    case property_verdictt::Failed:
      ++reached;
      label = "✓ REACHED";
      color = &GREEN;
      break;
    }

    log_status(
      "{}{}{}: '{}'{}",
      *color,
      label,
      RESET,
      prettify_solidity_expr(property),
      result.note.empty() ? "" : " (" + result.note + ")");
  }

  std::ostringstream oss;
  oss << "Coverage goals: " << verdicts.size() << " " << GREEN << "✓ "
      << reached << " reached" << RESET;
  if (unreached > 0)
    oss << ", - " << unreached << " unreached";
  if (undecided > 0)
    oss << ", " << YELLOW << "? " << undecided << " undecided" << RESET;

  log_result("{}", oss.str());
}

static void print_property_rows(
  const std::vector<property_rowt> &rows,
  const property_countst &counts,
  bool is_color)
{
  const std::string RESET = is_color ? "\033[0m" : "";

  log_result("\n** Results:");

  std::string group;
  for (const auto &row : rows)
  {
    const char *color = "";
    if (is_color)
      color = row.verdict == property_verdictt::Failed   ? "\033[31m"
              : row.verdict == property_verdictt::Passed ? "\033[32m"
                                                         : "\033[33m";

    const std::string row_group =
      row.file + (row.function.empty() ? "" : ", function " + row.function);
    if (row_group != group)
    {
      group = row_group;
      log_result("{}", group);
    }

    log_result(
      "  {}{:<11}{}  {:<{}}  line {:>{}}  {}{}",
      color,
      verdict_label(row.verdict),
      RESET,
      "[" + row.id + "]",
      counts.id_width,
      row.line,
      counts.line_width,
      prettify_solidity_expr(row.description),
      row.note.empty() ? "" : " (" + row.note + ")");
  }
}

static void print_property_summary(
  size_t total,
  const property_countst &counts,
  bool is_color)
{
  const std::string GREEN = is_color ? "\033[32m" : "";
  const std::string RED = is_color ? "\033[31m" : "";
  const std::string YELLOW = is_color ? "\033[33m" : "";
  const std::string RESET = is_color ? "\033[0m" : "";

  std::ostringstream oss;
  oss << "\n** " << (counts.failed > 0 ? RED : "") << counts.failed << " of "
      << total << " properties failed" << (counts.failed > 0 ? RESET : "");
  if (counts.passed > 0)
    oss << ", " << GREEN << counts.passed << " passed" << RESET;
  if (counts.unknown > 0)
    oss << ", " << YELLOW << counts.unknown << " unknown" << RESET;
  if (counts.not_checked > 0)
    oss << ", " << YELLOW << counts.not_checked << " not checked" << RESET;

  log_result("{}", oss.str());
}

/// Build, order and print one property table. \p unchecked_is_unknown renames
/// the rows a k-step run never decided: the run did check them, at every k it
/// reached, so "not checked" would understate what was tried.
static property_countst print_property_table(
  const std::map<std::string, property_resultt> &verdicts,
  const std::set<std::string> &library_files,
  bool is_color,
  bool unchecked_is_unknown,
  bool decided_only,
  const std::function<std::string(const expr2tc &)> &render_condition)
{
  std::vector<property_rowt> rows =
    build_property_rows(verdicts, library_files, render_condition);

  if (unchecked_is_unknown)
    for (auto &row : rows)
      if (row.verdict == property_verdictt::NotChecked)
        row.verdict = property_verdictt::Unknown;

  const property_countst counts = count_properties(rows);

  // An intermediate phase of an iterative strategy that decided nothing has
  // nothing to report; another phase will.
  if (decided_only && !counts.anything_decided())
    return counts;

  print_property_rows(rows, counts, is_color);
  print_property_summary(rows.size(), counts, is_color);
  return counts;
}

void report_k_step_property_table(
  const optionst &options,
  const goto_functionst &goto_functions,
  const namespacet &ns)
{
  if (!strategy_owns_property_table(options))
    return;

  const std::map<std::string, property_resultt> verdicts =
    goto_functionst::property_verdicts.snapshot();
  if (verdicts.empty())
    return;

  // Same caveat as the bmct constructor's: under Python a hidden body is not
  // ESBMC's own model, so nothing is demoted to the bottom of the table.
  print_property_table(
    verdicts,
    config.language.lid == language_idt::PYTHON
      ? std::set<std::string>()
      : collect_library_assertion_files(goto_functions),
    options.get_bool_option("color"),
    true,
    false,
    [&ns](const expr2tc &e) { return from_expr(ns, "", e); });

  if (goto_functionst::property_verdicts.is_incomplete())
    print_partial_report_note();
}

void bmct::report_property_verdicts(smt_resultt res) const
{
  const bool final = reports_final_verdict(res);

  // --dead-code-check turns --multi-property on implicitly. Its live-branch
  // probes reach a verdict of Failed, which contradicts both the [Dead code]
  // report and the advisory's forced SUCCESSFUL verdict (issue #4495).
  if (options.get_bool_option("dead-code-check"))
    return;

  const std::map<std::string, property_resultt> verdicts =
    goto_functionst::property_verdicts.snapshot();

  if (verdicts.empty())
    return;

  if (options.get_bool_option("coverage-measurement"))
  {
    // A path-coverage query names each claim's verdict as it is solved.
    if (!goto_coveraget::path_cov_query_mode())
      report_coverage_goal_verdicts(verdicts);
    return;
  }

  const property_countst counts = print_property_table(
    verdicts,
    library_files,
    options.get_bool_option("color"),
    strategy_owns_property_table(options),
    !final,
    [this](const expr2tc &e) { return from_expr(ns, "", e); });

  if (!final && !counts.anything_decided())
    return;

  const size_t failed = counts.failed;
  const size_t not_checked = counts.not_checked;

  // A violation was found but pinned on nothing: the solver answered sat
  // without a model to attribute it with (the subprocess SMT-LIB backends
  // under --result-only). Saying so beats a table that reads as contradicting
  // the verdict below it. A violation downgraded to unknown because it lies
  // downstream of a loop-invariant havoc is attributed to its own row, so it
  // is not this case (issue #7480).
  if (res == P_SATISFIABLE && failed == 0 && !weak_invariant_detected)
    log_result(
      "   (a violation exists, but this solver produced no model, so which "
      "property it violates could not be determined)");
  // Without --multi-property the run encodes one formula and stops at the
  // first violation, so it cannot separate the properties it never decided.
  // Say which flag would decide them rather than leave the gap unexplained.
  else if (not_checked > 0 && !options.get_bool_option("multi-property"))
    log_result(
      "   (this mode stops at the first violation; use --multi-property for a "
      "verdict on every property)");

  // Every property may have been discharged during symbolic execution, in
  // which case no solver ever ran and there is nothing to time. Average over
  // the properties that reached a verdict, not over the whole table: the
  // never-checked ones cost the solver nothing and would dilute it.
  const size_t decided = counts.passed + counts.failed + counts.unknown;
  if (!solver_stats.name.empty() && decided > 0)
  {
    std::ostringstream timing_oss;
    timing_oss << "Solver: " << solver_stats.name
               << " • Decision procedure total time: "
               << time2string(solver_stats.total_time_ms) << "s"
               << " • Avg: "
               << time2string(solver_stats.total_time_ms / decided)
               << "s/property";
    log_result("{}", timing_oss.str());
  }

  // The phase that prints a k-step run's table need not be the phase that
  // stopped short, so the store carries that across phases too.
  if (report_incomplete || goto_functionst::property_verdicts.is_incomplete())
    print_partial_report_note();
}
