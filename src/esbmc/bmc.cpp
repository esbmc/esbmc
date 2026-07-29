#include <csignal>
#include <memory>
#include <sys/types.h>
#include <algorithm>
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

#include <fmt/format.h>
#include <regex>
#include <ac_config.h>
#include <esbmc/bmc.h>
#include <esbmc/document_subgoals.h>
#include <fstream>
#include <goto-programs/goto_loops.h>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/features.h>
#include <goto-symex/xml_goto_trace.h>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <sstream>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/location.h>

#include <util/migrate.h>
#include <util/show_symbol_table.h>
#include <util/time_stopping.h>
#include <util/cache.h>
#include <atomic>
#include <nlohmann/json.hpp>

std::unordered_set<std::string> goto_functionst::reached_claims;
std::unordered_multiset<std::string> goto_functionst::reached_mul_claims;
std::mutex goto_functionst::reached_claims_mutex;
std::mutex goto_functionst::reached_mul_claims_mutex;
std::set<std::string> goto_functionst::truncated_loops;
std::mutex goto_functionst::truncated_loops_mutex;

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
        std::make_unique<assertion_cache>(config.ssa_caching_db));

    if (opts.get_bool_option("no-slice"))
      algorithms.emplace_back(std::make_unique<simple_slice>());
    else
      algorithms.emplace_back(std::make_unique<symex_slicet>(options));

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

void bmct::successful_trace(const symex_target_equationt &eq [[maybe_unused]])
{
  if (options.get_bool_option("result-only"))
    return;

  std::string witness_graphml_output =
    options.get_option("witness-output-graphml");
  std::string witness_yaml_output = options.get_option("witness-output-yaml");

  goto_tracet goto_trace;
  // correctness witness, why did goto trace ignore it in the past?
  // build_successful_goto_trace(eq, ns, goto_trace);
  if (witness_graphml_output != "")
    correctness_graphml_goto_trace(options, ns, goto_trace);

  if (witness_yaml_output != "")
    correctness_yaml_goto_trace(options, ns, goto_trace);
}

void bmct::error_trace(smt_convt &smt_conv, const symex_target_equationt &eq)
{
  if (options.get_bool_option("result-only"))
    return;

  log_progress("Building error trace");

  bool is_compact_trace = true;
  if (
    options.get_bool_option("no-slice") &&
    !options.get_bool_option("compact-trace"))
    is_compact_trace = false;

  goto_tracet goto_trace;
  build_goto_trace(eq, smt_conv, goto_trace, is_compact_trace);

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
    pytest_gen.generate_single(pytest_filename, eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate_single(".", eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-foundry-testcase"))
  {
    foundry_gen.generate_single(eq, smt_conv, ns);
  }

  if (options.get_bool_option("generate-html-report"))
    generate_html_report("1", ns, goto_trace, options);

  if (options.get_bool_option("generate-json-report"))
    generate_json_report("1", ns, goto_trace);

  // esbmc-minimise consumes this structured violation summary to seed
  // its mandatory set and oracle tuple.
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

smt_convt::resultt bmct::run_decision_procedure(
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
      return smt_convt::P_SMTLIB;
  }

  log_progress("Solving with solver {}", smt_conv.solver_text());

  fine_timet sat_start = current_time();
  smt_convt::resultt dec_result = smt_conv.dec_solve();
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

void bmct::show_program(const symex_target_equationt &eq)
{
  unsigned int count = 1;
  std::ostringstream oss;
  if (config.options.get_bool_option("ssa-symbol-table"))
    ::show_symbol_table_plain(ns, oss);

  languagest languages(ns, language_idt::C);

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

void bmct::report_trace(
  smt_convt::resultt &res,
  const symex_target_equationt &eq)
{
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");
  bool show_cex = options.get_bool_option("show-cex");

  switch (res)
  {
  case smt_convt::P_UNSATISFIABLE:
    if (is && term)
    {
    }
    else if (!bs)
    {
      successful_trace(eq);
    }
    break;

  case smt_convt::P_SATISFIABLE:
    if (!bs && show_cex)
    {
      error_trace(*runtime_solver, eq);
    }
    else if (!is && !fc)
    {
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
  for (auto &func : symex->goto_functions.function_map)
  {
    for (auto &instr : func.second.body.instructions)
    {
      std::lock_guard lock(instr.clear_claims_mutex);
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

void bmct::report_multi_property_trace(
  const smt_convt::resultt &res,
  const std::vector<witness_recordt> &witnesses,
  enumeration_stop_reasont stop_reason,
  const std::string &msg)
{
  if (options.get_bool_option("result-only"))
    return;

  switch (res)
  {
  case smt_convt::P_UNSATISFIABLE:
    log_success("Claim '{}' holds up to the current K", msg);
    return;

  case smt_convt::P_SATISFIABLE:
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
    log_fail("\n[Counterexample]\n");
    if (!witnesses.empty())
      show_goto_trace(oss, ns, witnesses.front().trace);
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
  oss << "\n[Counterexamples - " << witnesses.size() << " witnesses]\n\n";
  for (size_t i = 0; i < witnesses.size(); ++i)
  {
    const witness_recordt &w = witnesses[i];
    oss << "  ┌─ Witness " << (i + 1) << " of " << witnesses.size()
        << " ─────────────────────────────\n";
    oss << "  │  Inputs : ";
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
    oss << "  │  Trace  :\n";
    {
      std::ostringstream tr;
      show_goto_trace(tr, ns, w.trace);
      // Indent the trace under the box.
      std::string s = tr.str();
      std::string indented;
      indented.reserve(s.size() + 8);
      indented += "  │    ";
      for (char c : s)
      {
        indented += c;
        if (c == '\n')
          indented += "  │    ";
      }
      oss << indented << "\n";
    }
    oss << "  └──────────────────────────────────────────────\n\n";
  }

  oss << "Summary: " << witnesses.size()
      << " distinct input tuples violate this property (enumeration stopped: ";
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

  std::istringstream iss(loc);
  std::string token;
  while (iss >> token)
  {
    if (token == "file")
    {
      std::string val;
      iss >> val;
      j["file"] = val;
    }
    else if (token == "line")
    {
      int val = 0;
      iss >> val;
      j["line"] = val;
    }
    else if (token == "column")
    {
      int val = 0;
      iss >> val;
      j["column"] = val;
    }
    else if (token == "function")
    {
      std::string val;
      iss >> val;
      j["function"] = val;
    }
  }
  return j;
}

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims,
  pytest_generator &pytest_gen,
  ctest_generator &ctest_gen,
  foundry_generator &foundry_gen)
{
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

    log_success("\n[Coverage]\n");
    // The total assertion instances include the assert inside the source file, the unwinding asserts, the claims inserted during the goto-check and so on.
    log_result("Total Asserts: {}", total);
    if (total_instance >= tracked_instance)
      log_result("Total Assertion Instances: {}", total_instance);
    else
      // this could be
      // 1. the loop is too large that we cannot goto-unwind it
      // 2. the loop is somewhat non-deterministic that we cannot run goto-unwind
      log_result("Total Assertion Instances: unknown / non-deterministic");
    log_result("Reached Assertion Instances: {}", tracked_instance);

    // show claims
    if (options.get_bool_option("assertion-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_mul_claims)
      {
        log_status("  {}", prettify_solidity_expr(claim));
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

    // reached claims:
    auto total_cond_assert_cpy = total_cond_assert;
    for (const auto &claim_pair : total_cond_assert)
    {
      std::string claim_msg = claim_pair.first;
      std::string claim_loc = claim_pair.second;
      std::string claim_sig = claim_msg + "\t" + claim_loc;
      if (reached_claims.count(claim_sig))
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
        reached_claims.erase(claim_sig);
        total_cond_assert_cpy.erase(claim_pair);

        // reversal: obtain !ass
        if (
          claim_msg[0] == '!' && claim_msg[1] == '(' && claim_msg.back() == ')')
          // e.g. !(a==1)
          claim_msg = claim_msg.substr(2, claim_msg.length() - 3);
        else
          claim_msg = "!(" + claim_msg + ")";
        std::string r_claim_sig = claim_msg + "\t" + claim_loc;

        if (reached_claims.count(r_claim_sig))
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
        reached_claims.erase(r_claim_sig);
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

    // Default (no --coverage-covered-set): unchanged — raw reached set
    // (this also included non-universe entries like unwinding
    // assertions, kept as-is to preserve existing pinned numbers).
    // With the cross-run covered-set, the numerator is the universe
    // edges EITHER witnessed this run OR already persisted — and is
    // intersected with the universe, which also drops that over-count.
    size_t tracked_instance = reached_claims.size();
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
    if (total == 0)
    {
      log_result("No branch detected");
    }
    else
    {
      log_result("Branches : {}", total);
      log_result("Reached : {}", tracked_instance);

      // show claims
      if (options.get_bool_option("branch-coverage-claims"))
      {
        // reached claims:
        for (const auto &claim : reached_claims)
          log_status("  {}", prettify_solidity_expr(claim));
      }

      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    }

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
    // this also included the non-unwinding-assertions
    // which is not what we want
    const size_t tracked_instance = reached_claims.size();
    log_success("\n[Coverage]\n");
    if (total == 0)
    {
      log_result("No branch detected");
    }
    else
    {
      log_result("Function Entry Points & Branches : {}", total);
      log_result("Reached : {}", tracked_instance);

      // show claims
      if (options.get_bool_option("branch-function-coverage-claims"))
      {
        // reached claims:
        for (const auto &claim : reached_claims)
          log_status("  {}", prettify_solidity_expr(claim));
      }

      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    }
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
    const auto &redundant = goto_coveraget::k_path_spanning_redundant;
    auto is_maximal = [&redundant](const std::string &claim_sig) {
      // claim_sig = "msg\tloc"; loc has no tabs, so rfind is robust if a
      // future emission path puts a tab in msg.
      const auto tab = claim_sig.rfind('\t');
      return redundant.count(
               {claim_sig.substr(0, tab), claim_sig.substr(tab + 1)}) == 0;
    };

    const size_t tracked_instance =
      std::count_if(reached_claims.begin(), reached_claims.end(), is_maximal);

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
    // Suppressed rather than annotated, on precedent: the path-distribution line
    // once mixed a per-round count into a structural measurement and printed
    // "1 path(s) total ... 0.33x" for a contract whose real figures are 8 and
    // 2.67x. It was documented in a comment and still got quoted, and a measured
    // result had to be retracted. A number that does not exist cannot be quoted;
    // a number with a disclaimer attached can.
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
        "CERTIFIED box makes them hold and would be counted as uncovered. The "
        "result of this run is the VERIFICATION SUCCESSFUL / FAILED verdict "
        "below, and on FAILED the counterexample input inside the box");
    }
    else
    {
      // Denominator = the no-skip static universe built by
      // solidity_path_coverage() (one entry per enumerated complete path), so a
      // covered-set skip never shrinks it. Numerator = universe paths EITHER
      // witnessed this run OR already persisted by an earlier round.
      const size_t total = goto_coveraget::all_claims.size();
      // Complete-path coverage keys its cross-run cover on the CONTENT-ADDRESSED
      // stable id, not on `covered_set` (which solidity_path_coverage()
      // deliberately leaves empty — see path_witnessed_earlier). Using
      // covered_set here reported every path carried over from an earlier round
      // as U: a path with a counterexample in hand, filed under "undecided".
      const bool cov_set_active = !goto_coveraget::path_covered_outpath.empty();
      size_t tracked_instance = 0;
      for (const auto &k : goto_coveraget::all_claims)
        if (
          goto_coveraget::path_witnessed_earlier(k) ||
          reached_claims.count(k.first + "\t" + k.second))
          ++tracked_instance;

      log_success("\n[Coverage]\n");
      if (total == 0)
        log_result("No complete path enumerated");
      else
      {
        log_result("Complete Paths : {}", total);
        log_result("Reached : {}", tracked_instance);
        log_result("Path Coverage: {}%", tracked_instance * 100.0 / total);
        // Exit-shape breakdown on stdout as well as in the JSON. The JSON is a
        // file, and the regression harness matches program OUTPUT only — without
        // this line the whole exit classification (and in particular the refusal
        // to call an undetermined exit "normal") cannot be regression-locked.
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
          // that can silently absorb an implementation defect has already done so
          // three times, and each time the broken case looked exactly like an
          // honest timeout.
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
            if (
              goto_coveraget::path_witnessed_earlier(k) ||
              reached_claims.count(sig))
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
      // with `--coverage-covered-set cov.json` produced no file at all, so every
      // round re-instrumented everything and the escalation story (round N spends
      // its budget only on paths still lacking a CE) did not work at all.
      //
      // Same defect shape as the entry-liveness audit before it was wired: a
      // function that is written, tested by eye, and never called. The function
      // itself selects only 'F' claims, so nothing but a real witness is ever
      // persisted.
      if (cov_set_active)
      {
        goto_coveraget::write_path_covered_set_atomic();
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
        const bool witnessed = covered || prior;

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
            claim_entry["ce_extraction"]["extcall_returns_unavailable_reason"] =
              "not implemented yet. The value an external call returns to the "
              "contract reaches the user's variable through a tuple-field "
              "extraction, which get_nondet_symbol does not traverse, so that "
              "trace step is skipped before classification. The "
              "_ESBMC_Nondet_Extcall_* symbols that ARE in the trace are the "
              "re-entry model's method-choice bits, not the returned value "
              "(measured: identical on two paths that disagree about it), so "
              "they are deliberately not reported here. An empty list means "
              "UNKNOWN, not 'this path performs no external call'";
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
        }
        else if (tri == "F")
          // Witnessed, but not by THIS run: the cross-run covered-set already
          // held it, so the path was not re-instrumented and no model was
          // produced here. Without this the reader sees an F with no inputs and
          // no post-state and cannot tell that from "the CE was empty".
          claim_entry["ce_extraction"]["payload_absent_reason"] =
            "path was already witnessed in an earlier round (covered-set) and "
            "therefore not re-instrumented this run; its counterexample values "
            "are in the report of the round that witnessed it";
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
      report["summary"]["bound"]["max_tx"] =
        max_tx.empty() ? "default" : max_tx;
      report["summary"]["bound"]["unwind"] =
        unwind_s.empty() ? "default" : unwind_s;
      report["summary"]["bound"]["kind"] =
        unbounded_run ? "unbounded" : "bounded";
      report["summary"]["bound"]["tx_exploration"] = tx_exploration;
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
  }

  // Generate pytest test case from collected data (for coverage mode)
  if (options.get_bool_option("generate-pytest-testcase"))
  {
    std::string input_file = options.get_option("input-file");
    std::string module_name = pytest_generator::extract_module_name(input_file);
    std::string pytest_filename =
      pytest_generator::generate_pytest_filename(module_name);
    pytest_gen.generate(pytest_filename);
  }

  // Generate CTest test cases from collected data (for coverage mode)
  if (options.get_bool_option("generate-ctest-testcase"))
  {
    ctest_gen.generate();
  }

  // Generate Foundry test cases from collected data (for coverage mode)
  if (options.get_bool_option("generate-foundry-testcase"))
  {
    foundry_gen.generate();
  }
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
      if (totals == 0)
      {
        log_result("No branch detected");
      }
      else
      {
        // show claims
        if (options.get_bool_option("branch-coverage-claims"))
        {
          // reached claims:
          for (const auto &claim : reached_claims)
            log_status("  {}", prettify_solidity_expr(claim));
        }
        log_result("Branch Coverage: {}%", tracked_instance * 100.0 / totals);
      }
    }
    else if (is_branch_func_cov)
    {
      size_t totals = goto_coveraget::total_func_branch;
      const int tracked_instance = reached_claims.size();
      if (totals == 0)
      {
        log_result("No branch detected");
      }
      else
      {
        // show claims
        if (options.get_bool_option("branch-function-coverage-claims"))
        {
          // reached claims:
          for (const auto &claim : reached_claims)
            log_status("  {}", prettify_solidity_expr(claim));
        }
        log_result(
          "Branch Function Coverage: {}%", tracked_instance * 100.0 / totals);
      }
    }
    else
    {
      log_error("Unsupported coverage metrics");
      abort();
    }
  }
}

void bmct::report_result(smt_convt::resultt &res)
{
  // k-induction prints its own messages
  if (options.get_bool_option("k-induction-parallel"))
    return;
  // Diagnostic pass: per-property results are already printed by
  // multi_property_check; suppress any global verdict from this level.
  if (options.get_bool_option("diagnose-unknown-properties"))
    return;

  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");
  bool mul = options.get_bool_option("multi-property");

  switch (res)
  {
  case smt_convt::P_UNSATISFIABLE:
    if (is && term)
    {
      report_failure();
    }
    else if (!bs || mul)
    {
      // Suppress spurious success when a violation was already found in a
      // previous k step (multi-property sequential k-induction).  The final
      // verdict is printed by do_bmc_strategy once the loop terminates.
      // Exception: assertion-coverage mode always reports success after
      // coverage analysis, regardless of any violations found.
      if (
        !options.get_bool_option("kind-violation-found") ||
        options.get_bool_option("assertion-coverage") ||
        options.get_bool_option("assertion-coverage-claims"))
        report_success();
    }
    else
    {
      log_status("No bug has been found in the base case");
    }
    break;

  case smt_convt::P_SATISFIABLE:
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

    // Return failure if we didn't actually check anything, we just emitted the
    // test information to an SMTLIB formatted file. Causes esbmc to quit
    // immediately (with no error reported)
  case smt_convt::P_SMTLIB:
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

smt_convt::resultt bmct::start_bmc()
{
  std::shared_ptr<symex_target_equationt> eq;
  smt_convt::resultt res = run(eq);
  if (!options.get_bool_option("multi-property"))
    // multi-property traces are output during the run(eq)
    report_trace(res, *eq);
  report_result(res);
  return res;
}

smt_convt::resultt bmct::run(std::shared_ptr<symex_target_equationt> &eq)
{
  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  if (options.get_bool_option("schedule"))
    return run_thread(eq);

  smt_convt::resultt res;
  do
  {
    if (++interleaving_number > 1)
      log_status("Thread interleavings {}", interleaving_number);

    // Clear the cache between thread interleavings to prevent
    // incorrect caching of assertions with different thread contexts
    if (!options.get_bool_option("no-cache-asserts"))
      config.ssa_caching_db.clear();

    fine_timet bmc_start = current_time();
    res = run_thread(eq);

    if (res == smt_convt::P_SATISFIABLE)
    {
      if (config.options.get_bool_option("smt-model"))
        runtime_solver->print_model();

      if (config.options.get_bool_option("bidirectional"))
        bidirectional_search(*runtime_solver, *eq);
    }

    if (res)
    {
      if (res == smt_convt::P_SATISFIABLE)
        ++interleaving_failed;

      if (!options.get_bool_option("all-runs"))
        return res;
    }
    fine_timet bmc_stop = current_time();

    log_status("BMC program time: {}s", time2string(bmc_stop - bmc_start));

    // Only run for one run
    if (options.get_bool_option("interactive-ileaves"))
      return res;

  } while (symex->setup_next_formula());

  if (options.get_bool_option("ltl"))
  {
    // So, what was the lowest value ltl outcome that we saw?
    if (ltl_results_seen[ltl_res_bad])
      log_result("Final lowest outcome: LTL_BAD");
    else if (ltl_results_seen[ltl_res_failing])
      log_result("Final lowest outcome: LTL_FAILING");
    else if (ltl_results_seen[ltl_res_succeeding])
      log_result("Final lowest outcome: LTL_SUCCEEDING");
    else if (ltl_results_seen[ltl_res_good])
      log_result("Final lowest outcome: LTL_GOOD");
    else
      log_warning("No LTL traces seen, apparently");
  }

  return interleaving_failed > 0 ? smt_convt::P_SATISFIABLE : res;
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
    if (ssait.is_assert() && smt_conv.l_get(ssait.cond_ast).is_false())
    {
      if (!ssait.loop_number)
        return;

      // Save the location of the failed assertion
      frames = ssait.stack_trace;
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
      renaming::renaming_levelt::get_original_name(new_lhs, symbol2t::level0);

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

smt_convt::resultt bmct::run_thread(std::shared_ptr<symex_target_equationt> &eq)
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
      eq->check_for_duplicate_assigns();

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

    if (
      options.get_bool_option("program-only") ||
      options.get_bool_option("program-too"))
      show_program(*eq);

    if (options.get_bool_option("program-only"))
      return smt_convt::P_SMTLIB;

    log_status(
      "Generated {} VCC(s), {} remaining after simplification ({} "
      "assignments)",
      solver_result.total_claims,
      remaining_asserts,
      BigInt(eq->SSA_steps.size()) - ignored);

    if (options.get_bool_option("document-subgoals"))
    {
      std::ostringstream oss;
      document_subgoals(*eq, oss);
      log_status("{}", oss.str());
      return smt_convt::P_SMTLIB;
    }

    if (options.get_bool_option("show-vcc"))
    {
      show_vcc(*eq);
      return smt_convt::P_SMTLIB;
    }

    if (solver_result.remaining_claims == 0)
    {
      if (options.get_bool_option("smt-formula-only"))
      {
        log_status(
          "No VCC remaining, no SMT formula will be generated for"
          " this program\n");
        return smt_convt::P_SMTLIB;
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

      return smt_convt::P_UNSATISFIABLE;
    }

    if (options.get_bool_option("ltl"))
    {
      int res = ltl_run_thread(*eq);
      if (res == -1)
        return smt_convt::P_SMTLIB;
      if (res < 0)
        return smt_convt::P_ERROR;
      // Record that we've seen this outcome; later decide what the least
      // outcome was.
      ltl_results_seen[res]++;
      return smt_convt::P_UNSATISFIABLE;
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
        *eq, solver_result.remaining_claims, *runtime_solver);

    return run_decision_procedure(*runtime_solver, *eq);
  }

  catch (std::string &error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch (const char *error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory\n");
    return smt_convt::P_ERROR;
  }
}

int bmct::ltl_run_thread(symex_target_equationt &equation) const
{
  /* LTL checking - first check for whether we have a negative prefix, then
   * the indeterminate ones. */
  using Type = std::pair<std::string_view, ltl_res>;
  static constexpr std::array seq = {
    Type{"LTL_BAD", ltl_res_bad},
    Type{"LTL_FAILING", ltl_res_failing},
    Type{"LTL_SUCCEEDING", ltl_res_succeeding},
  };

  for (const auto &[which, check] : seq)
  {
    size_t num_asserts = 0;

    /* Start by turning all assertions that aren't the sought prefix assertion
     * into skips. */
    for (auto &SSA_step : equation.SSA_steps)
      if (SSA_step.is_assert())
      {
        if (SSA_step.comment != which)
          SSA_step.type = goto_trace_stept::SKIP;
        else
          num_asserts++;
      }

    smt_convt::resultt solver_result = smt_convt::P_UNSATISFIABLE;
    log_status("Checking for {}", which);
    if (num_asserts != 0)
    {
      std::unique_ptr<smt_convt> smt_conv(create_solver("", ns, options));
      solver_result = run_decision_procedure(*smt_conv, equation);
      if (solver_result == smt_convt::P_SATISFIABLE)
        log_status("Found trace satisfying {}", which);
    }
    else
      log_warning("Couldn't find {} assertion", which);

    /* Turn skip steps back into assertions. */
    for (auto &SSA_step : equation.SSA_steps)
      if (SSA_step.is_skip())
        for (const auto &[which2, _] : seq)
          if (SSA_step.comment == which2)
          {
            SSA_step.type = goto_trace_stept::ASSERT;
            break;
          }

    switch (solver_result)
    {
    case smt_convt::P_SATISFIABLE:
      return check;
    case smt_convt::P_ERROR:
      return -2;
    case smt_convt::P_SMTLIB:
      return -1;
    case smt_convt::P_UNSATISFIABLE:
      continue;
    }
  }

  /* Otherwise, we just got a good prefix. */
  return ltl_res_good;
}

smt_convt::resultt bmct::multi_property_check(
  const symex_target_equationt &eq,
  size_t remaining_claims,
  smt_convt &runtime_solver)
{
  // Initial values
  smt_convt::resultt final_result = smt_convt::P_UNSATISFIABLE;
  std::mutex result_mutex;
  std::atomic<size_t> ce_counter{0};
  // Sequential default consumes `jobs` in iteration order; using a sorted
  // vector lets us solve user-source claims before c2goto/library claims so
  // multi-property doesn't burn the budget on spurious library-side dereference
  // / overflow checks ahead of the user's real bugs.
  std::vector<size_t> jobs;

  // Add summary tracking
  SimpleSummary summary;
  summary.simplified_properties = symex->get_cur_state().simplified_claims;
  summary.total_properties = remaining_claims + summary.simplified_properties;
  summary.passed_properties =
    summary.passed_properties + summary.simplified_properties;

  // For coverage info
  auto &reached_claims = symex->goto_functions.reached_claims;
  auto &reached_mul_claims = symex->goto_functions.reached_mul_claims;
  auto &reached_claims_mutex = symex->goto_functions.reached_claims_mutex;
  auto &reached_mul_claims_mutex =
    symex->goto_functions.reached_mul_claims_mutex;

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
  // "k-Path Cov" — keyed off the dedicated boolean (see line ~717
  // comment); needed in the is_goto_cov disjunction so the
  // claim_slicer reads the witness comment, matching the form stored
  // in goto_coveraget::all_claims.
  bool is_k_path_cov = options.get_bool_option("k-path-coverage-enabled");
  // "Solidity complete-path Cov" — goals carry enc(pi) in the comment, same
  // as k-path, so it must join the is_goto_cov disjunction below or
  // claim_slicer would read the negated-expr instead of the comment and
  // every JSON entry would show up uncovered.
  bool is_path_cov = options.get_bool_option("solidity-path-coverage-enabled");

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

  // For color output
  bool is_color = options.get_bool_option("color");
  const std::string YELLOW = is_color ? "\033[33m" : "";

  // TODO: This is the place to check a cache
  jobs.reserve(remaining_claims);
  for (size_t i = 1; i <= remaining_claims; i++)
    jobs.push_back(i);

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
   * - &final_result: if the current instance is SAT, then we known that the current k contains a bug
   *
   * Finally, this function is affected by the "multi-fail-fast" option, which makes this instance stop
   * if final_result is set to SAT
   */
  auto job_function = [this,
                       &eq,
                       &ce_counter,
                       &final_result,
                       &result_mutex,
                       &summary,
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
                       &is_keep_verified,
                       &is_fail_fast,
                       &fail_fast_limit,
                       &fail_fast_cnt,
                       &bs,
                       &fc,
                       &is,
                       &is_color,
                       &YELLOW,
                       &runtime_solver](const size_t &i) {
    //"multi-fail-fast n": stop after first n SATs found.
    if (is_fail_fast && fail_fast_cnt >= fail_fast_limit)
      return;

    // Since this is just a copy, we probably don't need a lock
    symex_target_equationt local_eq = eq;

    // Set up the current claim and disable slice info output.
    // `is_goto_cov` flips claim_slicer's `claim_msg` source: in goto-cov
    // modes the slicer reads the comment (the original witness/guard
    // text we stored in insert_assert); otherwise it reads the negated
    // assertion expression. k-path goals are stored the same way as
    // branch / condition goals, so they must be in this disjunction —
    // otherwise the claim_sig built at line ~1751 disagrees with the
    // form in goto_coveraget::all_claims and every JSON entry shows up
    // as uncovered even when reached_claims has the matching reached
    // signature (PR #4330 review).
    bool is_goto_cov = is_assert_cov || is_cond_cov || is_branch_cov ||
                       is_branch_func_cov || is_k_path_cov || is_path_cov;
    claim_slicer claim(i, false, is_goto_cov, ns);
    claim.run(local_eq.SSA_steps);

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
    {
      ++summary.skipped_properties;
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

    // Initialize a solver
    smt_convt *solver_ptr = &runtime_solver;
    std::unique_ptr<smt_convt> new_solver;
    if (!options.get_bool_option("smt-during-symex"))
    {
      new_solver = std::unique_ptr<smt_convt>(create_solver("", ns, options));
      solver_ptr = new_solver.get();
    }

    // Store solver name initially but not again
    std::call_once(summary.solver_name_flag, [&]() {
      summary.solver_name = solver_ptr->solver_text();
    });
    // In coverage mode, only report instrumented coverage claims
    bool is_cov_silent =
      is_goto_cov && claim.claim_property != "instrumented assertion";

    if (!is_cov_silent)
      log_status(
        "Solving claim '{}' with solver {}",
        prettify_solidity_expr(claim.claim_cstr),
        solver_ptr->solver_text());

    // Save current instance with timing
    fine_timet solve_start = current_time();
    smt_convt::resultt solver_result =
      run_decision_procedure(*solver_ptr, local_eq);
    fine_timet solve_stop = current_time();

    // Show colored result after solving
    const std::string GREEN = is_color ? "\033[32m" : "";
    const std::string RED = is_color ? "\033[31m" : "";
    const std::string RESET = is_color ? "\033[0m" : "";

    if (!is_cov_silent)
    {
      if (solver_result == smt_convt::P_UNSATISFIABLE)
      {
        // Claim passed - show in green
        log_status(
          "{}✓ PASSED{}: '{}'",
          GREEN,
          RESET,
          prettify_solidity_expr(claim.claim_cstr));
      }
      else if (solver_result == smt_convt::P_SATISFIABLE)
      {
        if (is)
          // Inductive step could not prove this claim - show in yellow
          log_status(
            "{}? UNKNOWN{}: '{}'",
            YELLOW,
            RESET,
            prettify_solidity_expr(claim.claim_cstr));
        else
          // Claim failed (counterexample found) - show in red
          log_status(
            "{}✗ FAILED{}: '{}'",
            RED,
            RESET,
            prettify_solidity_expr(claim.claim_cstr));
      }
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
      if (solver_result == smt_convt::P_SATISFIABLE)
        verdict = is ? 'U' : 'F';
      else if (solver_result == smt_convt::P_UNSATISFIABLE)
        verdict = 'P';
      else
        verdict = 'U';
      std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
      // 'F' is final: a witness stays valid, so a later phase's failure to
      // reprove must never downgrade it.
      auto it_o = goto_coveraget::claim_outcome.find(claim_sig);
      if (it_o == goto_coveraget::claim_outcome.end())
        goto_coveraget::claim_outcome.emplace(claim_sig, verdict);
      else if (it_o->second != 'F')
        it_o->second = verdict;
    }

    double solve_time_s = (solve_stop - solve_start);

    // Atomically update summary with timing and results
    double old_total_time_s = summary.total_time_s;
    double new_total_time_s;
    do
    {
      new_total_time_s = old_total_time_s + solve_time_s;
    } while (!summary.total_time_s.compare_exchange_weak(
      old_total_time_s, new_total_time_s));

    if (solver_result == smt_convt::P_SATISFIABLE)
    {
      if (is)
        summary.unknown_properties++;
      else
        summary.failed_properties++;
    }
    else if (solver_result == smt_convt::P_UNSATISFIABLE)
      summary.passed_properties++;

    // If an assertion instance is verified to be violated
    if (solver_result == smt_convt::P_SATISFIABLE)
    {
      // Inductive step SAT means unprovable (UNKNOWN), not a real
      // counterexample — skip trace generation and return early.
      if (is)
      {
        std::lock_guard lock(result_mutex);
        final_result = solver_result;
        return;
      }

      bool is_compact_trace = true;
      if (
        options.get_bool_option("no-slice") &&
        !options.get_bool_option("compact-trace"))
        is_compact_trace = false;
      // The path-coverage CE payload is harvested from THIS trace. A compact
      // trace drops every step flagged `hidden`, and that is exactly where the
      // EVM environment (msg./tx./block.) and the harness-side writes are
      // assigned — the report's `env` block would come back empty even though
      // the values are in the model. Building the full trace changes nothing
      // about what is solved; it only costs trace-construction time.
      if (is_path_cov && options.get_bool_option("cov-report-json"))
        is_compact_trace = false;

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
      const std::string cex_output = options.get_option("cex-output");
      const std::string graphml_path =
        options.get_option("witness-output-graphml");
      const std::string yaml_path = options.get_option("witness-output-yaml");
      const bool want_graphml = !graphml_path.empty();
      const bool want_yaml = !yaml_path.empty();
      const bool want_testcase = options.get_bool_option("generate-testcase");
      const bool want_html = options.get_bool_option("generate-html-report");
      const bool want_json = options.get_bool_option("generate-json-report");
      const bool want_pytest =
        options.get_bool_option("generate-pytest-testcase");
      const bool want_ctest =
        options.get_bool_option("generate-ctest-testcase");
      const bool want_foundry =
        options.get_bool_option("generate-foundry-testcase");

      // Emit testcase metadata once per claim (not once per witness).
      if (want_testcase)
        generate_testcase_metadata();

      // Drive enumeration with a separate variable so the original SAT
      // outcome stays in `solver_result` for downstream bookkeeping
      // (final_result, fail-fast counter, claim cleanup).
      smt_convt::resultt enum_result = solver_result;
      bool ctx_pushed = false;
      while (enum_result == smt_convt::P_SATISFIABLE)
      {
        witness_recordt w;
        build_goto_trace(local_eq, *solver_ptr, w.trace, is_compact_trace);
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

        // Emit machine-readable artifacts NOW, while this witness's solver
        // model is still live. After the next dec_solve(), the model is
        // either gone (UNSAT) or replaced by the next witness's values.
        if (!cex_output.empty())
        {
          std::ofstream out(fmt::format("{}-{}", w.ce_index, cex_output));
          show_goto_trace(out, ns, w.trace);
        }
        // For graphml/yaml the writer reads the path from `options`;
        // override per-witness so multiple witnesses don't overwrite the
        // same file (and so it's safe under --parallel-solving).
        if (want_graphml)
          violation_graphml_goto_trace(
            options,
            ns,
            w.trace,
            fmt::format("{}-{}", w.ce_index, graphml_path));
        if (want_yaml)
          violation_yaml_goto_trace(
            options, ns, w.trace, fmt::format("{}-{}", w.ce_index, yaml_path));
        if (want_testcase)
          generate_testcase(
            "testcase-" + std::to_string(w.ce_index) + ".xml",
            local_eq,
            *solver_ptr);
        if (want_html)
          generate_html_report(
            std::to_string(w.ce_index), ns, w.trace, options);
        if (want_json)
          generate_json_report(std::to_string(w.ce_index), ns, w.trace);
        if (want_pytest)
          pytest_gen.collect(local_eq, *solver_ptr);
        if (want_ctest)
          ctest_gen.collect(local_eq, *solver_ptr, ns);
        if (want_foundry)
          // Collect into the external strategy-level generator when threaded
          // (--k-induction), else the owned member (plain BMC).
          foundry().collect(local_eq, *solver_ptr, ns);

        // Solidity complete-path coverage: harvest this path's CE payload
        // (concrete inputs + post-state) NOW, while this witness's solver
        // model is live — after the next dec_solve() it is gone. Only the
        // FIRST witness of a claim is recorded: one CE per complete path is
        // what the report contracts for.
        if (is_path_cov && witnesses.empty())
        {
          goto_coveraget::path_ce_t ce;
          ce.sliced = !options.get_bool_option("no-slice");
          // Same condition the dispatch uses to set protect_ce_symbols: with
          // the JSON requested, the payload symbols are exempt from slicing, so
          // `sliced` alone must not be read as "the payload was cut down".
          ce.payload_symbols_protected =
            options.get_bool_option("cov-report-json");
          ce.compact_trace = is_compact_trace;
          ce.revert_pre_rollback = goto_coveraget::revert_paths.count(
                                     {claim.claim_msg, claim.claim_loc}) > 0;
          // Ordered so the emitted post-state is deterministic across runs.
          std::map<std::string, std::string> last_state;
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
            // Mapping / dynamic-array state: these do NOT live in the contract
            // object, so the `this->` test below never sees them and they would
            // be discarded as harness noise — leaving a reader to infer the
            // mapping was untouched. Recognise them by their contract-level id
            // and record them as state; when the model value is not a constant
            // (an infinite-array store), record the NAME so the write is still
            // visible rather than silently lost.
            const std::string bid = base_sym_id(st.lhs);
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
              // (External-call returns were already taken above, before the
              // contract-scope-store branch could discard them.)
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
                if (input_seen.insert(name).second)
                  ce.env.emplace_back(name, val);
              }
              else
                ++ce.dropped_internal;
            }
          }
          for (const auto &[n, v] : last_state)
            ce.final_state.emplace_back(n, v);
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
          std::lock_guard lock(goto_coveraget::claim_outcome_mutex);
          goto_coveraget::path_ce[claim_sig] = std::move(ce);
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
        enum_result != smt_convt::P_UNSATISFIABLE &&
        enum_result != smt_convt::P_SATISFIABLE)
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
        if (is_path_cov && !goto_coveraget::path_covered_outpath.empty())
          goto_coveraget::write_path_covered_set_atomic();
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
          reached_claims,
          reached_mul_claims);
      else if (!is_cov_silent)
      {
        report_multi_property_trace(
          smt_convt::P_SATISFIABLE, witnesses, stop_reason, claim.claim_msg);
      }

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
    else if (solver_result == smt_convt::P_UNSATISFIABLE)
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

  // show summary
  report_simple_summary(summary);

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

void bmct::report_simple_summary(const SimpleSummary &summary) const
{
  if (options.get_bool_option("result-only"))
    return;

  // ANSI color codes
  bool is_color = options.get_bool_option("color");
  const std::string GREEN = is_color ? "\033[32m" : "";
  const std::string RED = is_color ? "\033[31m" : "";
  const std::string RESET = is_color ? "\033[0m" : "";

  // Build the properties summary string with colors
  std::ostringstream properties_oss;
  properties_oss << "Properties: " << summary.total_properties << " verified";

  if (summary.passed_properties > 0)
    properties_oss << " " << GREEN << "✓ " << summary.passed_properties
                   << " passed" << RESET;

  if (summary.skipped_properties > 0)
    properties_oss << ", " << GREEN << "✓ " << summary.skipped_properties
                   << " skipped" << RESET;

  if (summary.failed_properties > 0)
    properties_oss << ", " << RED << "✗ " << summary.failed_properties
                   << " failed" << RESET;

  if (summary.unknown_properties > 0)
  {
    const std::string YELLOW = is_color ? "\033[33m" : "";
    properties_oss << ", " << YELLOW << "? " << summary.unknown_properties
                   << " unknown" << RESET;
  }

  // Build the timing summary string
  double avg_time = summary.total_properties > 0
                      ? summary.total_time_s / summary.total_properties
                      : 0.0;

  std::ostringstream timing_oss;
  timing_oss << "Solver: " << summary.solver_name
             << " • Decision procedure total time: "
             << time2string(summary.total_time_s) << "s"
             << " • Avg: " << std::fixed << std::setprecision(1)
             << time2string(avg_time) << "s/property";

  // Output the summary
  log_result("{}", properties_oss.str());
  log_result("{}", timing_oss.str());
}