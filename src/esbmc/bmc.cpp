#include <csignal>
#include <memory>
#ifdef _WIN32
#  include <windows.h>
#endif
#include <sys/types.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
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
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/symex_invariant.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/features.h>
#include <goto-symex/sarif.h>
#include <goto-symex/symex_symmetry.h>
#include <goto-symex/xml_goto_trace.h>
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

bmct::bmct(goto_functionst &funcs, optionst &opts, contextt &_context)
  : options(opts), context(_context), ns(context)
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

void bmct::record_violated_properties(
  smt_convt &smt_conv,
  const symex_target_equationt &eq) const
{
  // A subprocess SMT-LIB backend answers sat/unsat without necessarily being
  // able to produce a model (that is what --result-only buys there). Which
  // claim failed is then genuinely unknown, so leave the properties
  // NotChecked rather than guess -- and do not ask, because get-value would
  // have nothing to read.
  if (!smt_conv.has_model())
    return;

  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assert() || step.ignore)
      continue;

    // Same idiom as build_goto_trace: an unevaluatable condition renders as
    // violated, not as held.
    if (smt_conv.l_get(step.cond_expr).is_true())
      continue;

    const locationt &location = step.source.pc->location;
    const std::string description = id2string(step.comment);
    goto_functionst::property_verdicts.record(
      description + " at " + location.as_string(),
      property_verdictt::Failed,
      property_location(location, description));
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
    pytest_gen.generate_single(
      pytest_output_dir(options), pytest_filename, eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate_single(ctest_output_dir(options), eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-html-report"))
    generate_html_report("1", ns, goto_trace, options);

  if (options.get_bool_option("generate-json-report"))
    generate_json_report("1", ns, goto_trace);

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

smt_resultt bmct::check_vacuity(symex_target_equationt &local_eq) const
{
  // Re-encode in vacuity mode: each kept assertion contributes its path
  // assumption to the OR'd disjunction instead of `not(assumpt -> claim)`.
  // The result is UNSAT iff the path to every kept claim is unreachable.
  std::unique_ptr<smt_convt> solver(create_solver("", ns, options));
  local_eq.convert(*solver, /*vacuity_mode=*/true);
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
    if (!bs && show_cex)
    {
      // --show-cex on an inductive-step or forward-condition SAT: that model
      // starts from a havoc'd state, so it witnesses no violation of the
      // program and must not reach a verdict (multi_property_check draws the
      // same line at its `is ? Unknown : Failed`).
      error_trace(*runtime_solver, eq);
    }
    else if (!is && !fc)
    {
      record_violated_properties(*runtime_solver, eq);
      error_trace(*runtime_solver, eq);
    }
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

      bool loc_match = (instr.location.as_string() == claim.claim_loc);
      bool expr_match = false;

      std::string guard_str = from_expr(ns, "", instr.guard);

      if (is_goto_cov)
        expr_match = (instr.location.comment().as_string() == claim.claim_msg);
      else
        expr_match = (guard_str == claim.claim_msg);

      if (loc_match && expr_match)
      {
        instr.make_skip();
      }
    }
  }
}

namespace
{
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

// Parse location string "file X line Y column Z function F" into components.
// A file path may contain spaces (util/location.cpp does not quote it), so a
// string-valued field is the run of words up to the next keyword rather than a
// single whitespace-delimited token.
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
// Reuses the branch-coverage instrumentation: a probe (an instrumented
// assertion over a branch guard) that multi_property_check never violated is
// unreachable under all inputs up to the current unwinding bound — i.e. that
// branch direction is dead. The dead set is therefore
// `all_claims \ reached_claims`. Findings are advisory: they are printed as a
// separate [Dead code] section and, when --sarif-output is set, emitted at
// SARIF note level. They never flip the verdict (see report_result).
static void report_dead_code(
  const optionst &options,
  const std::unordered_set<std::string> &reached_claims,
  const std::vector<dead_store_advisoryt> &dead_stores)
{
  std::vector<dead_code_finding_t> findings;

  for (const auto &[comment, loc] : goto_coveraget::all_claims)
  {
    const std::string claim_sig = comment + "\t" + loc;
    if (reached_claims.count(claim_sig))
      continue; // reachable branch direction — live code

    nlohmann::json parsed = parse_claim_location(loc);
    dead_code_finding_t f;
    f.file = parsed["file"].get<std::string>();
    f.line = static_cast<unsigned>(parsed["line"].get<int>());
    f.message = comment.empty()
                  ? "dead code: unreachable branch"
                  : "dead code: unreachable branch [guard: " + comment + "]";
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

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims,
  pytest_generator &pytest_gen,
  ctest_generator &ctest_gen)
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
    // The total assertion instances include the assert inside the source file, the unwinding asserts, the claims inserted during the goto-check and so on.
    // "Total/Unreached Asserts" count static claims; the "Instances" lines
    // count their goto-unwound copies.
    log_result("Total Asserts: {}", total);
    log_result("Unreached Asserts: {}", unreached_claims.size());
    if (total_instance >= tracked_instance)
      log_result("Total Assertion Instances: {}", total_instance);
    else
    {
      // this could be
      // 1. the loop is too large that we cannot goto-unwind it
      // 2. the loop is somewhat non-deterministic that we cannot run goto-unwind
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
    // this also included the non-unwinding-assertions
    // which is not what we want
    const size_t tracked_instance = reached_claims.size();
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
  }

  else if (is_branch_func_cov)
  {
    //! Might got incorrect total number when using --k-induction
    //! due to that the symex->goto_functions has been simplified
    const size_t total = goto_coveraget::total_func_branch;
    // this also included the non-unwinding-assertions
    // which is not what we want
    const size_t tracked_instance = reached_claims.size();
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
    else if (is_cond_cov)
      cov_type = "condition";
    else if (is_assert_cov)
      cov_type = "assertion";

    const auto &all_claims = goto_coveraget::all_claims;
    std::set<std::string> source_files;
    json claims_json = json::array();

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
      claims_json.push_back(claim_entry);
    }

    size_t total = all_claims.size();
    size_t covered_count = 0;
    for (const auto &c : claims_json)
      if (c["status"] == "covered")
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
    report["coverage_type"] = cov_type;
    report["source_files"] = json::array();
    for (const auto &f : source_files)
      report["source_files"].push_back(f);
    report["claims"] = claims_json;
    report["summary"]["total"] = total;
    report["summary"]["covered"] = covered_count;
    report["summary"]["uncovered"] = total - covered_count;
    report["summary"]["percentage"] =
      total > 0 ? covered_count * 100.0 / total : 0.0;

    std::ofstream out("cov-report.json");
    out << report.dump(2) << std::endl;
    log_success("Coverage report written to cov-report.json");
  }

  cov_block_reported = true;

  // Generate pytest test case from collected data (for coverage mode)
  if (options.get_bool_option("generate-pytest-testcase"))
  {
    std::string input_file = options.get_option("input-file");
    std::string module_name = pytest_generator::extract_module_name(input_file);
    std::string pytest_filename =
      pytest_generator::generate_pytest_filename(module_name);
    pytest_gen.generate(pytest_output_dir(options), pytest_filename);
  }

  // Generate CTest test cases from collected data (for coverage mode)
  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate(ctest_output_dir(options));
  }
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

// Output coverage information whenever an instrumented assertion is found violated.
// It is helpful when the program is too large and ESBMC cannot finish, we can still get some info about the coverage
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
  // total_cond: the combination of assertion's guard and location, which is used to identify each assertion in multi-property checking.

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
  if (options.get_bool_option("coverage-measurement"))
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
  bool mul = options.get_bool_option("multi-property");

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
        else if (vacuity_detected || ltl_uninstrumented)
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
      report_failure();
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
  // that is never cleared between interleavings, so by this point it holds every
  // probe reached by any of them. Emitting before report_result keeps the
  // [Dead code] section above the verdict, and routing the dead-store advisories
  // through the same call keeps both sets in one SARIF document (issue #4495).
  // Only a run that actually solved the probes can say anything about dead
  // code. --show-vcc returns P_SMTLIB from run_thread before
  // multi_property_check ever runs, and a solver failure gives P_ERROR; either
  // way reached_claims is empty while all_claims is full, so every branch would
  // be reported dead. report_result already declines to claim success over those
  // two results — stay silent here for the same reason.
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

  // A single monolithic UNSAT refutes the disjunction of every claim's
  // violation, so on a genuinely conclusive run each claim holds. Anything
  // weaker leaves the properties this phase never separated out as NotChecked.
  if (all_properties_proved(res))
    goto_functionst::property_verdicts.promote_unchecked_to_passed();

  // The report describes the whole run, so an iterative strategy prints it
  // with its final verdict rather than once per k. --multi-property has
  // always reported per phase, and keeps doing so where a phase decided
  // something.
  if (reports_final_verdict(res) || options.get_bool_option("multi-property"))
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

  return res;
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

smt_resultt bmct::run(std::shared_ptr<symex_target_equationt> &eq)
{
  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  const bool multi_property = options.get_bool_option("multi-property");
  if (multi_property)
    goto_functionst::property_verdicts.clear();
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
      // emission still stops immediately: those are not "live probe" results and
      // must propagate. It also leaves violation_seen clear, so the barren
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
      if (options.get_bool_option("multi-property"))
      {
        std::unordered_set<std::string> empty_reached;
        std::unordered_multiset<std::string> empty_mul_reached;
        pytest_generator empty_pytest;
        ctest_generator empty_ctest;
        report_coverage(
          options, empty_reached, empty_mul_reached, empty_pytest, empty_ctest);
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
          it->type = goto_trace_stept::SKIP;
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
      it->type = goto_trace_stept::ASSERT;

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

smt_resultt bmct::multi_property_check(
  const symex_target_equationt &eq,
  size_t remaining_claims,
  smt_convt &runtime_solver,
  unsigned int truncated_loops)
{
  // Initial values
  smt_resultt final_result = P_UNSATISFIABLE;
  std::mutex result_mutex;
  // Solved in claim order: an unordered container would make the per-claim
  // solve order — and which claim a shared-solver bug lands on — vary by
  // standard library.
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
    note_cov_incomplete(fmt::format(
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

  // is_vb: enable verbose output coverage info if the option "--verbosity coverage:N" is set, where N should larger than 0
  // By enabling this, we will output the coverage information when handling each instrumentation assertion.
  bool is_vb = messaget::state.modules["coverage"] != VerbosityLevel::None;

  // For incr/kind in multi-property
  bool is_keep_verified = options.get_bool_option("keep-verified-claims");
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");

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
  for (size_t i = 1; i <= remaining_claims; i++)
    jobs.push_back(i);

  /* This is a JOB that will:
   * 1. Generate a solver instance for a specific claim (@parameter i)
   * 2. Solve the instance
   * 3. Generate a Counter-Example (or Witness)
   *
   * This job also affects the environment by using:
   * - &ce_counter: for generating the Counter Example file name
   * - &final_result: if the current instance is SAT, then we known that the current k contains a bug
   *
   * Finally, this function is affected by the "multi-fail-fast" option, which makes this instance stop
   * if final_result is set to SAT
   */
  auto job_function = [this,
                       &eq,
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
                       &is_dead_code,
                       &is_cov_run,
                       &is_keep_verified,
                       &is_fail_fast,
                       &fail_fast_limit,
                       &fail_fast_cnt,
                       &bs,
                       &fc,
                       &is,
                       &runtime_solver](const size_t &i) {
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

    const property_locationt claim_ploc =
      property_location(claim.claim_location, claim.claim_comment);

    if (fail_fast_hit)
    {
      // The skipped probes were never solved. Counting them as unreached
      // would report a percentage that looks measured but is not.
      if (claim.claim_property == "instrumented assertion")
      {
        goto_functionst::property_verdicts.record(
          claim.claim_cstr, property_verdictt::Unknown, claim_ploc);
        note_undecided_cov_goal("--multi-fail-fast limit reached");
      }
      return;
    }

    // Drop claims that verified to be failed
    // we use the "comment + location" to distinguish each claim
    // to avoid double verifying the claims that are already verified
    //! This algo is unsound, need a better signature to distinguish claims
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
      std::lock_guard lock(reached_claims_mutex);
      is_verified = reached_claims.count(claim.claim_cstr) ? true : false;
    }
    if (is_assert_cov && is_verified)
    {
      // insert to the multiset before skipping the verification process
      std::lock_guard lock(reached_mul_claims_mutex);
      reached_mul_claims.emplace(claim_sig);
    }

    // skip if we have already verified
    if (is_verified && !is_keep_verified)
      return;

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
    // In coverage mode, only report instrumented coverage claims. Dead-code
    // detection is advisory: silence every per-claim solve/trace so only the
    // final [Dead code] summary is shown (issue #4495).
    bool is_cov_silent =
      is_goto_cov &&
      (is_dead_code || claim.claim_property != "instrumented assertion");
    // A coverage probe: SAT means "this location is reachable". It is not a
    // property, so it must not be reported as one (issue #6387). Keyed off
    // is_cov_run so a --dead-code-check probe keeps its own handling.
    const bool is_cov_goal =
      is_cov_run && claim.claim_property == "instrumented assertion";

    if (!is_cov_silent)
      log_status(
        "Solving claim '{}' with solver {}",
        prettify_solidity_expr(claim.claim_cstr),
        solver_ptr->solver_text());

    // Save current instance with timing
    fine_timet solve_start = current_time();
    smt_resultt solver_result = run_decision_procedure(*solver_ptr, local_eq);
    fine_timet solve_stop = current_time();

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
        goto_functionst::property_verdicts.record(
          claim.claim_cstr,
          is_vacuous ? property_verdictt::Unknown : property_verdictt::Passed,
          claim_ploc,
          is_vacuous ? "vacuous discharge: path assumptions are unsatisfiable; "
                       "possible causes include an over-constrained loop "
                       "invariant, requires clause, or upstream assume"
                     : "");
      else if (solver_result == P_SATISFIABLE)
        goto_functionst::property_verdicts.record(
          claim.claim_cstr,
          is ? property_verdictt::Unknown : property_verdictt::Failed,
          claim_ploc,
          is ? "inductive step could not prove this claim" : "");
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
            claim.claim_cstr, property_verdictt::Unknown, claim_ploc);
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
        build_goto_trace(local_eq, *solver_ptr, w.trace);
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
          reached_claims.emplace(claim.claim_cstr);
      }

      // for verbose output of cond coverage
      if (is_vb)
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
      // checked against the assertions the instrumentation replaced.
      if (!is_goto_cov)
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
      // either forward condition or inductive step
      if (!is_keep_verified && !bs)
      {
        clear_verified_claims_in_ssa(local_eq, claim, is_goto_cov);
        clear_verified_claims_in_goto(claim, is_goto_cov);
      }
  };

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
      parallel_jobs.push_back(std::thread(job_function, i));

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

  // For coverage with fixed bound unwinding
  if (
    bs && !fc && !is && !options.get_bool_option("k-induction") &&
    !options.get_bool_option("incremental-bmc"))
    report_coverage(
      options, reached_claims, reached_mul_claims, pytest_gen, ctest_gen);

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

  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assert())
      continue;

    const locationt &location = step.source.pc->location;
    const std::string description = id2string(step.comment);
    goto_functionst::property_verdicts.record(
      description + " at " + location.as_string(),
      property_verdictt::NotChecked,
      property_location(location, description));
  }
}

bool bmct::reports_final_verdict(smt_resultt res) const
{
  if (options.get_bool_option("k-induction-parallel"))
    return false;

  const bool fc = options.get_bool_option("forward-condition");
  const bool is = options.get_bool_option("inductive-step");

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
    "kind-violation-found",
    "suppress-bounded-success"};
  for (const char *option : disqualifying)
    if (options.get_bool_option(option))
      return false;

  const bool is = options.get_bool_option("inductive-step");
  if (
    is && (options.get_bool_option("termination") ||
           options.get_bool_option("disable-inductive-step")))
    return false;

  // A base case alone is bounded: it refutes a bug up to k, it does not prove
  // the property. Only multi-property records a per-claim verdict there.
  return !options.get_bool_option("base-case") ||
         options.get_bool_option("multi-property");
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

void bmct::print_property_rows(
  const std::vector<property_rowt> &rows,
  const property_countst &counts) const
{
  const bool is_color = options.get_bool_option("color");
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

void bmct::print_property_summary(size_t total, const property_countst &counts)
  const
{
  const bool is_color = options.get_bool_option("color");
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
    report_coverage_goal_verdicts(verdicts);
    return;
  }

  const std::vector<property_rowt> rows =
    build_property_rows(verdicts, library_files);
  const property_countst counts = count_properties(rows);

  // An intermediate phase of an iterative strategy that decided nothing has
  // nothing to report; another phase will.
  if (!final && !counts.anything_decided())
    return;

  print_property_rows(rows, counts);
  print_property_summary(rows.size(), counts);

  const size_t failed = counts.failed;
  const size_t not_checked = counts.not_checked;

  // A violation was found but pinned on nothing: the solver answered sat
  // without a model to attribute it with (the subprocess SMT-LIB backends
  // under --result-only). Saying so beats a table that reads as contradicting
  // the verdict below it.
  if (res == P_SATISFIABLE && failed == 0)
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

  if (report_incomplete)
    log_result(
      "This report is partial: the run stopped before every property reached "
      "a verdict, so properties are missing above, and a passing verdict "
      "holds only for the thread interleavings explored. Raise "
      "--multi-property-interleavings, or drop --multi-fail-fast, to check "
      "further.");
}
