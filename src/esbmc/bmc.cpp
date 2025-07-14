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
#include <goto-symex/witnesses.h>

std::unordered_set<std::string> goto_functionst::reached_claims;
std::unordered_multiset<std::string> goto_functionst::reached_mul_claims;
std::unordered_set<std::string> goto_functionst::verified_claims;

std::mutex goto_functionst::reached_claims_mutex;
std::mutex goto_functionst::reached_mul_claims_mutex;
std::mutex goto_functionst::verified_claims_mutex;

bmct::bmct(goto_functionst &funcs, optionst &opts, contextt &_context)
  : options(opts), context(_context), ns(context)
{
  interleaving_number = 0;
  interleaving_failed = 0;

  ltl_results_seen[ltl_res_bad] = 0;
  ltl_results_seen[ltl_res_failing] = 0;
  ltl_results_seen[ltl_res_succeeding] = 0;
  ltl_results_seen[ltl_res_good] = 0;

  // The next block will initialize the algorithms used for the analysis.
  {
    if (opts.get_bool_option("no-slice"))
      algorithms.emplace_back(std::make_unique<simple_slice>());
    else
      algorithms.emplace_back(std::make_unique<symex_slicet>(options));

    // Run cache if user has specified the option
    if (options.get_bool_option("cache-asserts"))
      // Store the set between runs
      algorithms.emplace_back(
        std::make_unique<assertion_cache>(
          config.ssa_caching_db,
          !options.get_bool_option("forward-condition")));

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

void bmct::successful_trace()
{
  if (options.get_bool_option("result-only"))
    return;

  std::string witness_output = options.get_option("witness-output");
  if (witness_output != "")
  {
    goto_tracet goto_trace;
    log_progress("Building successful trace");
    /* build_successful_goto_trace(eq, ns, goto_trace); */
    correctness_graphml_goto_trace(options, ns, goto_trace);
  }
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

  std::string witness_output = options.get_option("witness-output");
  if (witness_output != "")
    violation_graphml_goto_trace(options, ns, goto_trace);

  if (options.get_bool_option("generate-testcase"))
  {
    generate_testcase_metadata();
    generate_testcase("testcase.xml", eq, smt_conv);
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
    smt_conv.dump_smt();
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
      successful_trace();
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
  const std::unique_ptr<smt_convt> &solver,
  const symex_target_equationt &local_eq,
  const std::atomic<size_t> ce_counter,
  const goto_tracet &goto_trace,
  const std::string &msg)
{
  if (options.get_bool_option("result-only"))
    return;

  switch (res)
  {
  case smt_convt::P_UNSATISFIABLE:
    // UNSAT means that the property was correct up to K
    log_success("Claim '{}' holds up to the current K", msg);
    break;

  case smt_convt::P_SATISFIABLE:
  {
    std::string output_file = options.get_option("cex-output");
    if (output_file != "")
    {
      std::ofstream out(fmt::format("{}-{}", ce_counter.load(), output_file));
      show_goto_trace(out, ns, goto_trace);
    }

    if (options.get_bool_option("generate-testcase"))
    {
      generate_testcase_metadata();
      generate_testcase(
        "testcase-" + std::to_string(ce_counter) + ".xml", local_eq, *solver);
    }
    if (options.get_bool_option("generate-html-report"))
      generate_html_report(std::to_string(ce_counter), ns, goto_trace, options);

    if (options.get_bool_option("generate-json-report"))
      generate_json_report(std::to_string(ce_counter), ns, goto_trace);

    std::ostringstream oss;
    log_fail("\n[Counterexample]\n");
    show_goto_trace(oss, ns, goto_trace);
    log_result("{}", oss.str());
    break;
  }

  default:
    log_fail("Claim '{}' could not be solved", msg);
    break;
  }
}

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims)
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

  if (is_assert_cov)
  {
    const int total = goto_coveraget::total_assert;
    const int tracked_instance = reached_mul_claims.size();
    const int total_instance = goto_coveraget::total_assert_ins;

    if (total)
    {
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
    }

    // show claims
    if (options.get_bool_option("assertion-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_mul_claims)
      {
        log_status("  {}", claim);
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
          log_status("  {} : SATISFIED", claim_sig);

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
            log_result("  {} : SATISFIED", r_claim_sig);
        }
        else
        {
          ++unsat_instance;
          if (cond_show_claims)
            log_result("  {} : UNSATISFIED", r_claim_sig);
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
        log_result("  {}", claim_sig);
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
    if (total)
    {
      log_success("\n[Coverage]\n");
      // The total assertion instances include the assert inside the source file, the unwinding asserts, the claims inserted during the goto-check and so on.
      log_result("Branches : {}", total);
      log_result("Reached : {}", tracked_instance);
    }

    // show claims
    if (options.get_bool_option("branch-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_claims)
        log_status("  {}", claim);
    }

    if (total != 0)
      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    else
      log_result("Branch Coverage: 0%");
  }

  else if (is_branch_func_cov)
  {
    //! Might got incorrect total number when using --k-induction
    //! due to that the symex->goto_functions has been simplified
    const size_t total = goto_coveraget::total_func_branch;
    // this also included the non-unwinding-assertions
    // which is not what we want
    const size_t tracked_instance = reached_claims.size();
    if (total)
    {
      log_success("\n[Coverage]\n");
      // The total assertion instances include the assert inside the source file, the unwinding asserts, the claims inserted during the goto-check and so on.
      log_result("Function Entry Points & Branches : {}", total);
      log_result("Reached : {}", tracked_instance);
    }

    // show claims
    if (options.get_bool_option("branch-function-coverage-claims"))
    {
      // reached claims:
      for (const auto &claim : reached_claims)
        log_status("  {}", claim);
    }

    if (total != 0)
      log_result("Branch Coverage: {}%", tracked_instance * 100.0 / total);
    else
      log_result("Branch Coverage: 0%");
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
        log_status("\n  {} : SATISFIED", claim_sig);
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
          log_status("  {}", claim);
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
          log_status("  {}", claim);
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
          log_status("  {}", claim);
      }

      if (totals != 0)
        log_result(
          "Branch Function Coverage: {}%", tracked_instance * 100.0 / totals);
      else
        log_result("Branch Function Coverage: 0%");
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

    if (
      options.get_bool_option("program-only") ||
      options.get_bool_option("program-too"))
      show_program(*eq);

    if (options.get_bool_option("program-only"))
      return smt_convt::P_SMTLIB;

    log_status(
      "Generated {} VCC(s), {} remaining after simplification ({} assignments)",
      solver_result.total_claims,
      solver_result.remaining_claims,
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
      options.get_bool_option("base-case"))
      return multi_property_check(*eq, solver_result.remaining_claims);

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
  size_t remaining_claims)
{
  // As of now, it only makes sense to do this for the base-case
  assert(
    options.get_bool_option("base-case") &&
    "Multi-property only supports base-case");

  // Initial values
  smt_convt::resultt final_result = smt_convt::P_UNSATISFIABLE;
  std::mutex result_mutex;
  std::atomic<size_t> ce_counter{0};
  std::unordered_set<size_t> jobs;

  // Add summary tracking
  SimpleSummary summary;
  summary.total_properties = remaining_claims;

  // For coverage info
  auto &reached_claims = symex->goto_functions.reached_claims;
  auto &reached_mul_claims = symex->goto_functions.reached_mul_claims;
  auto &verified_claims = symex->goto_functions.verified_claims;
  auto &reached_claims_mutex = symex->goto_functions.reached_claims_mutex;
  auto &reached_mul_claims_mutex =
    symex->goto_functions.reached_mul_claims_mutex;
  auto &verified_claims_mutex = symex->goto_functions.verified_claims_mutex;

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

  // TODO: This is the place to check a cache
  for (size_t i = 1; i <= remaining_claims; i++)
    jobs.emplace(i);

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
                       &verified_claims,
                       &reached_claims_mutex,
                       &reached_mul_claims_mutex,
                       &verified_claims_mutex,
                       &is_assert_cov,
                       &is_cond_cov,
                       &is_vb,
                       &is_branch_cov,
                       &is_branch_func_cov,
                       &is_keep_verified,
                       &is_fail_fast,
                       &fail_fast_limit,
                       &fail_fast_cnt,
                       &bs,
                       &fc,
                       &is,
                       &is_color](const size_t &i)
  {
    //"multi-fail-fast n": stop after first n SATs found.
    if (is_fail_fast && fail_fast_cnt >= fail_fast_limit)
      return;

    // Since this is just a copy, we probably don't need a lock
    symex_target_equationt local_eq = eq;

    // Set up the current claim and disable slice info output
    bool is_goto_cov =
      is_assert_cov || is_cond_cov || is_branch_cov || is_branch_func_cov;
    claim_slicer claim(i, false, is_goto_cov, ns);
    claim.run(local_eq.SSA_steps);

    // Drop claims that verified to be failed
    // we use the "comment + location" to distinguish each claim
    // to avoid double verifying the claims that are already verified
    //! This algo is unsound, need a better signature to distinguish claims
    bool is_verified = false;
    std::string claim_sig = claim.claim_msg + "\t" + claim.claim_loc;
    if (is_assert_cov)
      // C++20 reached_mul_claims.contains
      is_verified = reached_mul_claims.count(claim_sig) ? true : false;
    else
      is_verified = reached_claims.count(claim_sig) ? true : false;
    if (is_assert_cov && is_verified)
    {
      // insert to the multiset before skipping the verification process
      std::lock_guard lock(reached_mul_claims_mutex);
      reached_mul_claims.emplace(claim_sig);
    }

    if (verified_claims.count(claim_sig))
    {
      clear_verified_claims_in_ssa(local_eq, claim, is_goto_cov);
      clear_verified_claims_in_goto(claim, is_goto_cov);
      is_verified = true;
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
    std::unique_ptr<smt_convt> runtime_solver(create_solver("", ns, options));

    // Store solver name initially but not again
    std::call_once(
      summary.solver_name_flag,
      [&]() { summary.solver_name = runtime_solver->solver_text(); });

    log_status(
      "Solving claim '{}' with solver {}",
      claim.claim_msg,
      runtime_solver->solver_text());

    // Save current instance with timing
    fine_timet solve_start = current_time();
    smt_convt::resultt solver_result =
      run_decision_procedure(*runtime_solver, local_eq);
    fine_timet solve_stop = current_time();

    // Show colored result after solving
    const std::string GREEN = is_color ? "\033[32m" : "";
    const std::string RED = is_color ? "\033[31m" : "";
    const std::string RESET = is_color ? "\033[0m" : "";

    if (solver_result == smt_convt::P_UNSATISFIABLE)
    {
      // Claim passed - show in green
      log_status("{}✓ PASSED{}: '{}'", GREEN, RESET, claim.claim_cstr);
    }
    else if (solver_result == smt_convt::P_SATISFIABLE)
    {
      // Claim failed - show in red
      log_status("{}✗ FAILED{}: '{}'", RED, RESET, claim.claim_cstr);
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
      summary.failed_properties++;
    else if (solver_result == smt_convt::P_UNSATISFIABLE)
      summary.passed_properties++;

    // If an assertion instance is verified to be violated
    if (solver_result == smt_convt::P_SATISFIABLE)
    {
      bool is_compact_trace = true;
      if (
        options.get_bool_option("no-slice") &&
        !options.get_bool_option("compact-trace"))
        is_compact_trace = false;

      goto_tracet goto_trace;
      build_goto_trace(local_eq, *runtime_solver, goto_trace, is_compact_trace);

      // Store claim_sig
      if (is_assert_cov)
      {
        std::lock_guard lock(reached_mul_claims_mutex);
        reached_mul_claims.emplace(claim_sig);
      }
      else
      {
        std::lock_guard lock(reached_claims_mutex);
        reached_claims.emplace(claim_sig);
      }

      // update cex number
      size_t previous_ce_counter;
      previous_ce_counter = ce_counter++;

      // for verbose output of cond coverage
      if (is_vb)
        report_coverage_verbose(
          claim,
          claim_sig,
          is_assert_cov,
          is_cond_cov,
          is_branch_cov,
          is_branch_func_cov,
          reached_claims,
          reached_mul_claims);
      else
      {
        report_multi_property_trace(
          solver_result,
          runtime_solver,
          local_eq,
          previous_ce_counter,
          goto_trace,
          claim.claim_msg);
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
    report_coverage(options, reached_claims, reached_mul_claims);

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
