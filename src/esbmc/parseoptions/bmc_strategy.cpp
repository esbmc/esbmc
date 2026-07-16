#include <ac_config.h>

#ifndef _WIN32
extern "C"
{
#  include <fcntl.h>
#  include <unistd.h>

#  ifdef HAVE_SENDFILE_ESBMC
#    include <sys/sendfile.h>
#  endif

#  include <sys/resource.h>
#  include <sys/time.h>
#  include <sys/types.h>
}
#endif

#include <esbmc/bmc.h>
#include <esbmc/esbmc_parseoptions.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/sarif.h>
#include <util/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config.h>
#include <util/filesystem.h>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <util/expr_util.h>
#include <iostream>
#include <fstream>
#include <goto-programs/add_race_assertions.h>
#include <goto-programs/add_restrict_assertions.h>
#include <goto-programs/goto_atomicity_check.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_termination.h>
#include <esbmc/ranking_synthesis.h>
#include <esbmc/non_termination.h>
#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/gcse.h>
#include <goto-programs/loop_numbers.h>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/write_goto_binary.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_exceptions.h>
#include <goto-programs/set_claims.h>
#include <goto-programs/show_claims.h>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/goto_check_uninit_vars.h>
#include <goto-programs/goto_check_unchecked_return.h>
#include <goto-programs/dead_store_analysis.h>
#include <goto-programs/mark_decl_as_non_det.h>
#include <goto-programs/assign_params_as_non_det.h>
#include <goto2c/goto2c.h>
#include <util/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/symbol.h>
#include <util/time_stopping.h>
#include <goto-programs/goto_cfg.h>
#include <langapi/language_util.h>
#include <goto-programs/contracts/contracts.h>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <fcntl.h>
#  ifdef __GLIBC__
#    include <execinfo.h>
#  endif
#endif

#ifdef ENABLE_GOTO_CONTRACTOR
#  include <goto-programs/goto_contractor.h>
#endif

// Emit CWE-835 (non-termination / infinite loop) into the structured
// outputs when --termination refutes the termination property.
//
// A non-termination verdict is proven by UNSAT at a loop exit marker, so
// unlike a normal property violation it has no counterexample trace to
// drive the structured emitters. We synthesise a single-step trace
// anchored to a real termination-marker ASSERT instruction: that gives a
// valid `pc` and a source location, and lets the existing emitters reuse
// the util/cwe_mapping single source of truth (@p comment matches the
// "non-terminating execution" rule -> CWE-835).
//
// CWE-835 is a loop weakness, so a per-loop marker is preferred as the
// anchor; an abort-call marker is only a fallback. Markers in library
// helpers (body.hide) rank below both, so the anchor never lands in
// ESBMC's own installed sources. The inductive-step verdict is a
// whole-program property, so with several loops present the chosen
// marker is a best-effort representative location — only the CWE id
// (835) is guaranteed, not that it is the specific non-terminating loop.
static void report_non_termination_cwe(
  optionst &options,
  const namespacet &ns,
  const goto_functionst &goto_functions,
  const std::string &comment)
{
  // The YAML (SV-COMP 2.0) witness format has no CWE field — CWE support is
  // GraphML-only — so it is deliberately not emitted here.
  const bool want_sarif = !options.get_option("sarif-output").empty();
  const bool want_graphml =
    !options.get_option("witness-output-graphml").empty();
  const bool want_json = options.get_bool_option("generate-json-report");
  if (!(want_sarif || want_graphml || want_json))
    return;

  // Anchor candidates, best first. A user-source location always beats a
  // library helper: goto_termination inserts per-loop markers into helpers
  // too (see the marker pass there), and __ESBMC_atexit_handler's
  // `while (atexit_count > 0)` is linked into every program. Since
  // function_map is ordered by mangled id, `c:@F@__ESBMC_atexit_handler`
  // sorts before `c:@F@main`, so scanning without this rank anchors the
  // CWE to ESBMC's own stdlib.c instead of the user's loop.
  enum anchor_rankt
  {
    USER_LOOP = 0,
    USER_ABORT,
    LIB_LOOP,
    LIB_ABORT,
    NO_ANCHOR
  };

  goto_programt::const_targett marker;
  anchor_rankt best = NO_ANCHOR;
  for (const auto &fn : goto_functions.function_map)
  {
    if (!fn.second.body_available)
      continue;
    const bool is_lib = fn.second.body.hide;
    for (auto it = fn.second.body.instructions.begin();
         best != USER_LOOP && it != fn.second.body.instructions.end();
         ++it)
    {
      if (!it->is_assert())
        continue;
      const std::string mc = it->location.comment().as_string();
      anchor_rankt rank;
      if (mc == "termination per-loop marker")
        rank = is_lib ? LIB_LOOP : USER_LOOP;
      else if (mc == "termination abort-call marker")
        rank = is_lib ? LIB_ABORT : USER_ABORT;
      else
        continue;
      if (rank < best)
      {
        marker = it;
        best = rank;
      }
    }
    if (best == USER_LOOP)
      break;
  }
  if (best == NO_ANCHOR)
    return;

  goto_tracet trace;
  goto_trace_stept step;
  step.step_nr = 1;
  step.thread_nr = 0;
  step.type = goto_trace_stept::ASSERT;
  step.guard = false; // a violated assert
  step.pc = marker;
  step.comment = comment;
  trace.steps.push_back(step);

  if (want_graphml)
    violation_graphml_goto_trace(options, ns, trace);
  if (want_json)
    generate_json_report("1", ns, trace);
  if (want_sarif)
    sarif_goto_trace(options, ns, trace);
}

// This method iteratively applies one of the verification strategies
// for different unwinding bounds up to the specified maximum depth.
//
// ESBMC features 4 verification strategies:
//
//  1) Incremental
//  2) Termination
//  3) Falsification
//  4) k-induction
//
// Applying a strategy in this context means solving a particular sequence
// of decision problems from the list below for the given unwinding bound k:
//
//  - Base case             (see "is_base_case_violated")
//  - Forward condition     (see "does_forward_condition_hold")
//  - Inductive step        (see "is_inductive_step_violated")
//
// \param options - options for setting the verification strategy
// and controlling symbolic execution
// \param goto_functions - GOTO program under verification
int esbmc_parseoptionst::do_bmc_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  // Get max number of iterations
  uint64_t max_k_step = cmdline.isset("unlimited-k-steps")
                          ? std::numeric_limits<uint64_t>::max()
                          : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  // Get the start of the base-case, default 1
  unsigned k_step_base = strtoul(cmdline.getval("base-k-step"), nullptr, 10);

  // For pytest test generation
  pytest_generator pytest_gen;

  // For ctest test generation
  ctest_generator ctest_gen;

  if (k_step_base >= max_k_step)
  {
    log_error(
      "--base-k-step ({}) must be smaller than --max-k-step ({}).",
      k_step_base,
      max_k_step);
    abort();
  }

  // Track whether any violation was found across all k steps.
  // In multi-property mode the loop continues past a violation to check
  // remaining properties, so we must remember the failure for the final verdict.
  bool any_violation_found = false;

  // Helper: emit the final verdict and return the correct exit code once a
  // proof or refutation has been found.  In multi-property mode the loop may
  // have continued past an earlier violation, so we must return 1 even when
  // the closing step (FC/IS) itself succeeds.
  auto conclude = [&]() -> int {
    // In coverage mode violations are expected; always report success.
    if (any_violation_found && !is_coverage)
    {
      log_fail("\nVERIFICATION FAILED");
      return 1;
    }
    return 0;
  };

  // Trying all bounds from 1 to "max_k_step" in "k_step_inc"
  uint64_t last_k_step = k_step_base;
  for (uint64_t k_step = k_step_base; k_step <= max_k_step;
       k_step += k_step_inc)
  {
    last_k_step = k_step;
    // k-induction
    if (options.get_bool_option("k-induction"))
    {
      bool is_bcv =
        is_base_case_violated(options, goto_functions, k_step).is_true();
      if (is_bcv)
      {
        any_violation_found = true;
        // Suppress spurious VERIFICATION SUCCESSFUL from report_result at
        // subsequent k steps where no new violations are found.
        options.set_option("kind-violation-found", true);
      }

      if (
        is_bcv && !cmdline.isset("multi-property") &&
        !options.get_bool_option("multi-property"))
        return 1;

      // if the property is proven violated in the bs, it's unnecessary to further run fw and is
      // this will make the trace looks cleaner yet might lead to an extra round to terminate the verification
      if (
        !is_bcv &&
        does_forward_condition_hold(options, goto_functions, k_step).is_false())
      {
        if (is_coverage)
          report_coverage(
            options,
            goto_functions.reached_claims,
            goto_functions.reached_mul_claims,
            pytest_gen,
            ctest_gen);
        return conclude();
      }

      // Don't run inductive step for k_step == 1
      if (k_step > 1)
      {
        if (
          !is_bcv && is_inductive_step_violated(options, goto_functions, k_step)
                       .is_false())
        {
          if (is_coverage)
            report_coverage(
              options,
              goto_functions.reached_claims,
              goto_functions.reached_mul_claims,
              pytest_gen,
              ctest_gen);
          return conclude();
        }
      }
    }
    // termination
    if (options.get_bool_option("termination"))
    {
      // `assert(false)` was inserted after main() and every loop havoc'd
      // by goto_termination. Property: "all executions terminate".
      //
      //   - Forward condition UNSAT at k:
      //       All states up to depth k are reachable — loops have fully
      //       unwound within k iters. Universal termination proven.
      //       Property HOLDS → return 0.
      //
      //   - Inductive step UNSAT at k:
      //       From no havoc'd iterate can the program reach end-of-main
      //       within k iters. A non-terminating execution exists.
      //       Property REFUTED → return 1.
      //
      // IS SAT is NOT a success condition: it only witnesses one
      // terminating path from one havoc'd state, which doesn't prove
      // all paths terminate.
      //
      // IS UNSAT is only sound when the k-induction havoc actually
      // covered the loop variables. Under --add-symex-value-sets,
      // loops that only modify pointers are SKIPPED by the havoc
      // transform (see goto_k_induction.cpp:91-94), so the IS just
      // runs the concrete initial state forward. IS UNSAT then means
      // "loop hasn't exited within k iters from initial state" —
      // which says nothing about non-termination; the loop may simply
      // need more iters. Disable the IS non-termination signal in
      // that mode and rely on FC alone.
      //
      // A linear ranking function proved every loop terminates (checked
      // once, before the havoc transform). This is k-independent, so
      // report success immediately without unwinding.
      if (options.get_bool_option("termination-ranking-proved"))
      {
        log_success(
          "\nRanking function shows all executions terminate\n"
          "VERIFICATION SUCCESSFUL");
        return 0;
      }

      // A recurrent-set non-termination check found an inductive R such
      // that every reachable state under R has an input continuation
      // staying in R and avoiding all exits (Gupta et al., POPL 2008).
      // The loop is non-terminating; report FAILED without unwinding.
      if (options.get_bool_option("termination-non-termination-proved"))
      {
        const std::string comment =
          "Recurrent set shows a non-terminating execution";
        const namespacet ns(context);
        report_non_termination_cwe(options, ns, goto_functions, comment);
        const std::string cwes = format_cwe_list(cwe_for(comment));
        std::string msg = "\n" + comment + "\n";
        if (!cwes.empty())
          msg += "CWE: " + cwes + "\n";
        msg += "VERIFICATION FAILED";
        log_fail("{}", msg);
        return 1;
      }

      // Skip IS for k = 1 (degenerates to a base-case check).
      if (does_forward_condition_hold(options, goto_functions, k_step)
            .is_false())
      {
        log_result(
          "\nForward condition shows all executions terminate "
          "(k = {:d})",
          k_step);
        return 0;
      }

      // IS UNSAT is only sound when k-induction actually havoc'd every
      // loop the property depends on. goto_k_induction skips a loop
      // when its modified set is empty — in that case
      // disable-inductive-step gets set mid-symex by the function-
      // pointer / recursion / concurrency hooks and the IS verdict is
      // treated as inconclusive below. Pointer-modifying loops are
      // now sound under --add-symex-value-sets thanks to the
      // value-set assume in symex_dereference, so no extra structural
      // gate is needed.
      if (k_step > 1)
      {
        tvt is_res =
          is_inductive_step_violated(options, goto_functions, k_step);
        // Symex may have set disable-inductive-step mid-run (function
        // pointers, recursion, concurrency). The IS UNSAT result is
        // then a vacuous "0 VCCs to falsify" and not a real
        // non-termination witness. Treat it as inconclusive.
        if (
          is_res.is_false() &&
          !options.get_bool_option("disable-inductive-step"))
        {
          const std::string comment = fmt::format(
            "Inductive step shows a non-terminating execution (k = {})",
            k_step);
          const namespacet ns(context);
          report_non_termination_cwe(options, ns, goto_functions, comment);
          const std::string cwes = format_cwe_list(cwe_for(comment));
          std::string msg = "\n" + comment;
          if (!cwes.empty())
            msg += "\nCWE: " + cwes;
          log_result("{}", msg);
          return 1;
        }
        // IS SAT or UNKNOWN — inconclusive, try larger k.
      }
    }
    // incremental-bmc
    if (options.get_bool_option("incremental-bmc"))
    {
      bool is_bcv =
        is_base_case_violated(options, goto_functions, k_step).is_true();
      if (is_bcv)
      {
        any_violation_found = true;
        options.set_option("kind-violation-found", true);
      }

      if (
        is_bcv && !cmdline.isset("multi-property") &&
        !options.get_bool_option("multi-property"))
        return 1;

      if (
        !is_bcv &&
        does_forward_condition_hold(options, goto_functions, k_step).is_false())
      {
        if (is_coverage)
          report_coverage(
            options,
            goto_functions.reached_claims,
            goto_functions.reached_mul_claims,
            pytest_gen,
            ctest_gen);
        return conclude();
      }
    }
    // falsification
    if (options.get_bool_option("falsification"))
    {
      if (is_base_case_violated(options, goto_functions, k_step).is_true())
        return 1;
    }
  }

  if (
    options.get_bool_option("multi-property") &&
    options.get_bool_option("k-induction"))
    diagnose_unknown_properties(options, goto_functions, last_k_step);

  log_status("Unable to prove or falsify the program, giving up.");
  log_fail("VERIFICATION UNKNOWN");

  if (is_coverage)
    report_coverage(
      options,
      goto_functions.reached_claims,
      goto_functions.reached_mul_claims,
      pytest_gen,
      ctest_gen);
  return 0;
}

// This is a wrapper method that does a single round of
// symbolic execution of the given GOTO program and creates
// a decision problem specified by the verification options.
// In brief, they are used to control what assertions and
// assumptions are injected into the verified bounded trace
// during symbolic execution.
//
// \param bmc - the bmc object that contains all the necessary
// information (see below) to perform a single run of Bounded Model Checking:
//
//  1) GOTO program,
//  2) verification options.
//  3) program context,
int esbmc_parseoptionst::do_bmc(bmct &bmc)
{
  log_progress("Starting Bounded Model Checking");

  // Forward dead-store advisories (CWE-563) to bmc so they reach SARIF on both
  // the success and failure paths. Empty unless --dead-store-check is set.
  bmc.dead_store_advisories = dead_store_advisories;

  smt_resultt res;
  try
  {
    res = bmc.start_bmc();
  }
  catch (const inductive_step_disabled_exceptiont &e)
  {
    // Symex hit an IS-unsound construct (recursion, threads,
    // function-pointer call) and threw to short-circuit. Return
    // P_ERROR so the strategy layer drops to TV_UNKNOWN; the caller
    // also checks `disable-inductive-step` to suppress any verdict.
    log_status("Inductive step aborted: {}", e.reason);
    res = P_ERROR;
  }
  catch (const smtlib_convt::external_process_died &e)
  {
    // An external SMT solver process (an --smtlib solver, or a one-shot
    // backend's model solver) died or returned an unusable response past the
    // backend's own recovery — e.g. while a counterexample was being read out
    // via (get-value). Report a clean failure rather than let the exception
    // reach std::terminate.
    log_error("SMT solver process failed: {}", e.what());
    res = P_ERROR;
  }

#ifdef HAVE_SENDFILE_ESBMC
  if (bmc.options.get_bool_option("memstats"))
  {
    int fd = open("/proc/self/status", O_RDONLY);
    sendfile(2, fd, nullptr, 100000);
    close(fd);
  }
#endif

  return res;
}
