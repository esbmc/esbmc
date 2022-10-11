#include <csignal>
#include <memory>
#include <sys/types.h>
#include <algorithm>
#include <thread>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#else
#include <windows.h>
#include <winbase.h>
#undef ERROR
#undef small
#endif

#include <fmt/format.h>
#include <ac_config.h>
#include <esbmc/bmc.h>
#include <esbmc/document_subgoals.h>
#include <fstream>
#include <goto-programs/goto_loops.h>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/slice.h>
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

bmct::bmct(goto_functionst &funcs, optionst &opts, contextt &_context)
  : options(opts), context(_context), ns(context)
{
  interleaving_number = 0;
  interleaving_failed = 0;

  // The next block will initialize the algorithms used for the analysis.
  {
    if(opts.get_bool_option("no-slice"))
      algorithms.emplace_back(std::make_unique<simple_slice>());
    else
      algorithms.emplace_back(std::make_unique<symex_slicet>(options));

    // Run cache if user has specified the option
    if(options.get_bool_option("cache-asserts"))
      // Store the set between runs
      algorithms.emplace_back(std::make_unique<assertion_cache>(
        config.ssa_caching_db, !options.get_bool_option("forward-condition")));
  }

  if(options.get_bool_option("smt-during-symex"))
  {
    runtime_solver = std::shared_ptr<smt_convt>(create_solver("", ns, options));

    symex = std::make_shared<reachability_treet>(
      funcs,
      ns,
      options,
      std::shared_ptr<runtime_encoded_equationt>(
        new runtime_encoded_equationt(ns, *runtime_solver)),
      _context);
  }
  else
  {
    symex = std::make_shared<reachability_treet>(
      funcs,
      ns,
      options,
      std::shared_ptr<symex_target_equationt>(new symex_target_equationt(ns)),
      _context);
  }
}

void bmct::do_cbmc(
  std::shared_ptr<smt_convt> &smt_conv,
  std::shared_ptr<symex_target_equationt> &eq)
{
  eq->convert(*smt_conv.get());
}

void bmct::successful_trace()
{
  if(options.get_bool_option("result-only"))
    return;

  std::string witness_output = options.get_option("witness-output");
  if(witness_output != "")
  {
    goto_tracet goto_trace;
    log_status("Building successful trace");
    /* build_successful_goto_trace(eq, ns, goto_trace); */
    correctness_graphml_goto_trace(options, ns, goto_trace);
  }
}

void bmct::error_trace(
  std::shared_ptr<smt_convt> &smt_conv,
  std::shared_ptr<symex_target_equationt> &eq)
{
  if(options.get_bool_option("result-only"))
    return;

  log_status("Building error trace");

  bool is_compact_trace = true;
  if(
    options.get_bool_option("no-slice") &&
    !options.get_bool_option("compact-trace"))
    is_compact_trace = false;

  goto_tracet goto_trace;
  build_goto_trace(eq, smt_conv, goto_trace, is_compact_trace);

  std::string output_file = options.get_option("cex-output");
  if(output_file != "")
  {
    std::ofstream out(output_file);
    show_goto_trace(out, ns, goto_trace);
  }

  std::string witness_output = options.get_option("witness-output");
  if(witness_output != "")
    violation_graphml_goto_trace(options, ns, goto_trace);

  std::ostringstream oss;
  oss << "\nCounterexample:\n";
  show_goto_trace(oss, ns, goto_trace);
  log_result("{}", oss.str());
}

void bmct::generate_smt_from_equation(
  std::shared_ptr<smt_convt> &smt_conv,
  std::shared_ptr<symex_target_equationt> &eq)
{
  std::string logic;

  if(!options.get_bool_option("int-encoding"))
  {
    logic = "bit-vector";
    logic += (!config.ansi_c.use_fixed_for_float) ? "/floating-point " : " ";
    logic += "arithmetic";
  }
  else
    logic = "integer/real arithmetic";

  log_status("Encoding remaining VCC(s) using {}", logic);

  fine_timet encode_start = current_time();
  do_cbmc(smt_conv, eq);
  fine_timet encode_stop = current_time();
  log_status(
    "Encoding to solver time: {}s", time2string(encode_stop - encode_start));
}

smt_convt::resultt bmct::run_decision_procedure(
  std::shared_ptr<smt_convt> &smt_conv,
  std::shared_ptr<symex_target_equationt> &eq)
{
  generate_smt_from_equation(smt_conv, eq);

  if(
    options.get_bool_option("smt-formula-too") ||
    options.get_bool_option("smt-formula-only"))
  {
    smt_conv->dump_smt();
    if(options.get_bool_option("smt-formula-only"))
      return smt_convt::P_SMTLIB;
  }

  log_status("Solving with solver {}", smt_conv->solver_text());

  fine_timet sat_start = current_time();
  smt_convt::resultt dec_result = smt_conv->dec_solve();
  fine_timet sat_stop = current_time();

  // output runtime
  log_status(
    "Runtime decision procedure: {}s", time2string(sat_stop - sat_start));

  return dec_result;
}

void bmct::report_success()
{
  log_status("\nVERIFICATION SUCCESSFUL");
}

void bmct::report_failure()
{
  log_status("\nVERIFICATION FAILED");
}

void bmct::show_program(std::shared_ptr<symex_target_equationt> &eq)
{
  unsigned int count = 1;
  std::ostringstream oss;
  if(config.options.get_bool_option("ssa-symbol-table"))
    ::show_symbol_table_plain(ns, oss);

  languagest languages(ns, language_idt::C);

  oss << "\nProgram constraints: \n";

  bool sliced = config.options.get_bool_option("ssa-sliced");

  for(auto const &it : eq->SSA_steps)
  {
    if(!(it.is_assert() || it.is_assignment() || it.is_assume()))
      continue;

    if(it.ignore && !sliced)
      continue;

    oss << "// " << it.source.pc->location_number << " ";
    oss << it.source.pc->location.as_string();
    if(!it.comment.empty())
      oss << " (" << it.comment << ")";
    oss << "\n/* " << count << " */ ";

    std::string string_value;
    languages.from_expr(migrate_expr_back(it.cond), string_value);

    if(it.is_assignment())
    {
      oss << string_value << "\n";
    }
    else if(it.is_assert())
    {
      oss << "(assert)" << string_value << "\n";
    }
    else if(it.is_assume())
    {
      oss << "(assume)" << string_value << "\n";
    }
    else if(it.is_renumber())
    {
      oss << "renumber: " << from_expr(ns, "", it.lhs) << "\n";
    }

    if(!migrate_expr_back(it.guard).is_true())
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
  std::shared_ptr<symex_target_equationt> &eq)
{
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");
  bool show_cex = options.get_bool_option("show-cex");

  switch(res)
  {
  case smt_convt::P_UNSATISFIABLE:
    if(is && term)
    {
    }
    else if(!bs)
    {
      successful_trace();
    }
    break;

  case smt_convt::P_SATISFIABLE:
    if(!bs && show_cex)
    {
      error_trace(runtime_solver, eq);
    }
    else if(!is && !fc)
    {
      error_trace(runtime_solver, eq);
    }
    break;

  default:
    break;
  }
}

void bmct::report_multi_property_trace(
  smt_convt::resultt &res,
  const std::string &msg)
{
  assert(options.get_bool_option("base-case"));

  switch(res)
  {
  case smt_convt::P_UNSATISFIABLE:
    // UNSAT means that the property was correct up to K
    log_status("Claim '{}' holds up to the current K", msg);
    break;

  case smt_convt::P_SATISFIABLE:
    // SAT means that we found a error!
    log_status("Claim '{}' fails", msg);
    break;

  default:
    log_status("Claim '{}' could not be solved", msg);
    break;
  }
}

void bmct::report_result(smt_convt::resultt &res)
{
  bool bs = options.get_bool_option("base-case");
  bool fc = options.get_bool_option("forward-condition");
  bool is = options.get_bool_option("inductive-step");
  bool term = options.get_bool_option("termination");

  switch(res)
  {
  case smt_convt::P_UNSATISFIABLE:
    if(is && term)
    {
      report_failure();
    }
    else if(!bs)
    {
      report_success();
    }
    else
    {
      log_status("No bug has been found in the base case");
    }
    break;

  case smt_convt::P_SATISFIABLE:
    if(!is && !fc)
    {
      report_failure();
    }
    else if(fc)
    {
      log_status("The forward condition is unable to prove the property");
    }
    else if(is)
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

  if((interleaving_number > 0) && options.get_bool_option("all-runs"))
  {
    log_status(
      "Number of generated interleavings: " +
      integer2string((interleaving_number)));
    log_status(
      "Number of failed interleavings: " +
      integer2string((interleaving_failed)));
  }
}

smt_convt::resultt bmct::start_bmc()
{
  std::shared_ptr<symex_target_equationt> eq;
  smt_convt::resultt res = run(eq);
  report_trace(res, eq);
  report_result(res);
  return res;
}

smt_convt::resultt bmct::run(std::shared_ptr<symex_target_equationt> &eq)
{
  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  if(options.get_bool_option("schedule"))
    return run_thread(eq);

  smt_convt::resultt res;
  do
  {
    if(++interleaving_number > 1)
      log_status("Thread interleavings {}", interleaving_number);

    fine_timet bmc_start = current_time();
    res = run_thread(eq);

    if(res == smt_convt::P_SATISFIABLE)
    {
      if(config.options.get_bool_option("smt-model"))
        runtime_solver->print_model();

      if(config.options.get_bool_option("bidirectional"))
        bidirectional_search(runtime_solver, eq);
    }

    if(res)
    {
      if(res == smt_convt::P_SATISFIABLE)
        ++interleaving_failed;

      if(!options.get_bool_option("all-runs"))
        return res;
    }
    fine_timet bmc_stop = current_time();

    log_status("BMC program time: {}s", time2string(bmc_stop - bmc_start));

    // Only run for one run
    if(options.get_bool_option("interactive-ileaves"))
      return res;

  } while(symex->setup_next_formula());

  return interleaving_failed > 0 ? smt_convt::P_SATISFIABLE : res;
}

void bmct::bidirectional_search(
  std::shared_ptr<smt_convt> &smt_conv,
  std::shared_ptr<symex_target_equationt> &eq)
{
  // We should only analyse the inductive step's cex and we're running
  // in k-induction mode
  if(!(options.get_bool_option("inductive-step") &&
       options.get_bool_option("k-induction")))
    return;

  // We'll walk list of SSA steps and look for inductive assignments
  std::vector<stack_framet> frames;
  unsigned assert_loop_number = 0;
  for(auto ssait : eq->SSA_steps)
  {
    if(ssait.is_assert() && smt_conv->l_get(ssait.cond_ast).is_false())
    {
      if(!ssait.loop_number)
        return;

      // Save the location of the failed assertion
      frames = ssait.stack_trace;
      assert_loop_number = ssait.loop_number;

      // We are not interested in instructions before the failed assertion yet
      break;
    }
  }

  for(auto f : frames)
  {
    // Look for the function
    goto_functionst::function_mapt::iterator fit =
      symex->goto_functions.function_map.find(f.function);
    assert(fit != symex->goto_functions.function_map.end());

    // Find function loops
    goto_loopst loops(f.function, symex->goto_functions, fit->second);

    if(!loops.get_loops().size())
      continue;

    auto lit = loops.get_loops().begin(), lie = loops.get_loops().end();
    while(lit != lie)
    {
      auto loop_head = lit->get_original_loop_head();

      // Skip constraints from other loops
      if(loop_head->loop_number == assert_loop_number)
        break;

      ++lit;
    }

    if(lit == lie)
      continue;

    // Get the loop vars
    auto all_loop_vars = lit->get_modified_loop_vars();
    all_loop_vars.insert(
      lit->get_unmodified_loop_vars().begin(),
      lit->get_unmodified_loop_vars().end());

    // Now, walk the SSA and get the last value of each variable before the loop
    std::unordered_map<irep_idt, std::pair<expr2tc, expr2tc>, irep_id_hash>
      var_ssa_list;

    for(auto ssait : eq->SSA_steps)
    {
      if(ssait.loop_number == lit->get_original_loop_head()->loop_number)
        break;

      if(ssait.ignore)
        continue;

      if(!ssait.is_assignment())
        continue;

      expr2tc new_lhs = ssait.original_lhs;
      renaming::renaming_levelt::get_original_name(new_lhs, symbol2t::level0);

      if(all_loop_vars.find(new_lhs) == all_loop_vars.end())
        continue;

      var_ssa_list[to_symbol2t(new_lhs).thename] = {
        ssait.original_lhs, ssait.rhs};
    }

    if(!var_ssa_list.size())
      return;

    // Query the solver for the value of each variable
    std::vector<expr2tc> equalities;
    for(auto it : var_ssa_list)
    {
      // We don't support arrays or pointers
      if(is_array_type(it.second.first) || is_pointer_type(it.second.first))
        return;

      auto lhs = build_lhs(smt_conv, it.second.first);
      auto value = build_rhs(smt_conv, it.second.second);

      // Add lhs and rhs to the list of new constraints
      equalities.push_back(equality2tc(lhs, value));
    }

    // Build new assertion
    expr2tc constraints = equalities[0];
    for(std::size_t i = 1; i < equalities.size(); ++i)
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
  std::shared_ptr<goto_symext::symex_resultt> result;

  fine_timet symex_start = current_time();
  try
  {
    if(options.get_bool_option("schedule"))
    {
      result = symex->generate_schedule_formula();
    }
    else
    {
      result = symex->get_next_formula();
    }
  }

  catch(std::string &error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch(const char *error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch(std::bad_alloc &)
  {
    log_error("Out of memory\n");
    return smt_convt::P_ERROR;
  }

  fine_timet symex_stop = current_time();

  eq = std::dynamic_pointer_cast<symex_target_equationt>(result->target);

  log_status(
    "Symex completed in: {}s ({} assignments)",
    time2string(symex_stop - symex_start),
    eq->SSA_steps.size());

  if(options.get_bool_option("double-assign-check"))
    eq->check_for_duplicate_assigns();

  try
  {
    BigInt ignored;
    for(auto &a : algorithms)
    {
      a->run(eq->SSA_steps);
      ignored += a->ignored();
    }

    if(
      options.get_bool_option("program-only") ||
      options.get_bool_option("program-too"))
      show_program(eq);

    if(options.get_bool_option("program-only"))
      return smt_convt::P_SMTLIB;

    log_status(
      "Generated {} VCC(s), {} remaining after simplification ({} assignments)",
      result->total_claims,
      result->remaining_claims,
      BigInt(eq->SSA_steps.size()) - ignored);

    if(options.get_bool_option("document-subgoals"))
    {
      std::ostringstream oss;
      document_subgoals(*eq.get(), oss);
      log_status(oss.str());
      return smt_convt::P_SMTLIB;
    }

    if(options.get_bool_option("show-vcc"))
    {
      show_vcc(eq);
      return smt_convt::P_SMTLIB;
    }

    if(result->remaining_claims == 0)
    {
      if(options.get_bool_option("smt-formula-only"))
      {
        log_status(
          "No VCC remaining, no SMT formula will be generated for"
          " this program\n");
        return smt_convt::P_SMTLIB;
      }

      return smt_convt::P_UNSATISFIABLE;
    }

    if(!options.get_bool_option("smt-during-symex"))
    {
      runtime_solver =
        std::shared_ptr<smt_convt>(create_solver("", ns, options));
    }

    if(
      options.get_bool_option("multi-property") &&
      options.get_bool_option("base-case"))
      return multi_property_check(eq, result->remaining_claims);
    else
      return run_decision_procedure(runtime_solver, eq);
  }

  catch(std::string &error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch(const char *error_str)
  {
    log_error("{}", error_str);
    return smt_convt::P_ERROR;
  }

  catch(std::bad_alloc &)
  {
    log_error("Out of memory\n");
    return smt_convt::P_ERROR;
  }
}

smt_convt::resultt bmct::multi_property_check(
  std::shared_ptr<symex_target_equationt> &eq,
  size_t remaining_claims)
{
  // As of now, it only makes sense to do this for the base-case
  assert(
    options.get_bool_option("base-case") &&
    "Multi-property only supports base-case");

  // Initial values
  smt_convt::resultt final_result = smt_convt::P_UNSATISFIABLE;
  std::atomic_size_t ce_counter = 0;
  std::unordered_set<size_t> jobs;
  std::mutex result_mutex;

  // TODO: This is the place to check a cache
  for(size_t i = 1; i <= remaining_claims; i++)
    jobs.emplace(i);

  /* This is a JOB that will:
   * 1. Generate a solver instance for a specific claim (@parameter i)
   * 2. Solve the instance
   * 3. Generate a Counter-Example (or Witness)
   *
   * This job also affects the environment by using:
   * - &ce_counter: for generating the Counter Example file name
   * - &final_result: if the current instance is SAT, then we known that the current k contains a bug
   * - &result_mutex: a mutex for step 3.
   *
   * Finally, this function is affected by the "multi-fail-fast" option, which makes this instance stop
   * if final_result is set to SAT
   */
  auto job_function =
    [this, &eq, &ce_counter, &final_result, &result_mutex](const size_t &i) {
      // Since this is just a copy, we probably don't need a lock
      auto local_eq = std::make_shared<symex_target_equationt>(*eq);

    // Just to confirm that things are in parallel
#ifndef _WIN32
      log_debug("Thread running on Core {}", sched_getcpu());
#endif
      // Set up the current claim and slice it!
      claim_slicer claim(i);
      claim.run(local_eq->SSA_steps);
      symex_slicet slicer(options);
      slicer.run(local_eq->SSA_steps);

      // Initialize a solver
      auto runtime_solver =
        std::shared_ptr<smt_convt>(create_solver("", ns, options));
      // Save current instance
      generate_smt_from_equation(runtime_solver, local_eq);

      log_status(
        "Solving claim {} with solver {}",
        claim.claim_msg,
        runtime_solver->solver_text());

      smt_convt::resultt result;
      /* TODO: We might move this into solver_convt. It is
       * useful to have the solver as a thread.
       */
      std::thread solver_job(
        [&result, &runtime_solver]() { result = runtime_solver->dec_solve(); });

      const bool fail_fast = options.get_bool_option("multi-fail-fast");
      // This loop is mainly for fail-fast.
      try
      {
        while(!solver_job.joinable())
        {
          // Try again 100ms later
          using namespace std::chrono_literals;
          std::this_thread::sleep_for(100ms);
          // Did someone finished already?
          if(fail_fast && final_result == smt_convt::P_SATISFIABLE)
          {
            log_status("Other thread already found a SAT VCC.");
            throw 0;
          }
        }
        solver_job.join();
        report_multi_property_trace(result, claim.claim_msg);
        if(result == smt_convt::P_SATISFIABLE)
        {
          const std::lock_guard<std::mutex> lock(result_mutex);
          // First, check if someone else find the solution!
          if(fail_fast && final_result == smt_convt::P_SATISFIABLE)
          {
            log_status(
              "Found solution for VCC. But, other thread found it first.");
            throw 0;
          }
          goto_tracet goto_trace;
          build_goto_trace(local_eq, runtime_solver, goto_trace, false);
          // TODO: Replace this with a test-case for coverage!
          std::string output_file = options.get_option("cex-output");
          if(output_file != "")
          {
            std::ofstream out(fmt::format("{}-{}", output_file, ce_counter++));
            show_goto_trace(out, ns, goto_trace);
          }
          final_result = result;
        }
        // TODO: This is the place to store into a cache
      }
      catch(...)
      {
        log_status("Failing Fast");
      }
    };

  // PARALLEL
  if(options.get_bool_option("parallel-solving"))
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
    for(const auto &i : jobs)
      parallel_jobs.push_back(std::thread(job_function, i));

    // Main driver
    size_t not_finished = parallel_jobs.size();
    while(not_finished)
    {
      for(auto &t : parallel_jobs)
      {
        if(t.joinable())
        {
          t.join();
          // There is no data-race in this loop, we should be fine.
          not_finished--;
        }
      }
      // We could filter out parallel_jobs for the ones that were already joined.
      // However, its probably not worth for small vectors.
    }
  }
  // SEQUENTIAL
  else
    std::for_each(std::begin(jobs), std::end(jobs), job_function);

  // TODO: Add a proper report for the current K!
  return final_result;
}
