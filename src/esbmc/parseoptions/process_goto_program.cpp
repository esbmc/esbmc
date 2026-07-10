#include <ac_config.h>

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
#include <util/cwe_mapping.h>
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

// This method performs various analyses and transformations
// on the given GOTO program. They involve all the techniques that we class
// as "static analyses" - performed on the given GOTO program before it is
// symbolically executed. Examples of such techniques include:
//
//  - interval analysis,
//  - removal of unreachable code,
//  - preprocessing the program for k-induction,
//  - applying GOTO contractors,
//  - ...
//
// \param options - various options used by the processing methods,
// \param goto_functions - reference to the GOTO program to be processed.
bool esbmc_parseoptionst::process_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    namespacet ns(context);

    bool is_mul =
      cmdline.isset("multi-property") || cmdline.isset("parallel-solving");
    is_coverage = cmdline.isset("assertion-coverage") ||
                  cmdline.isset("assertion-coverage-claims") ||
                  cmdline.isset("condition-coverage") ||
                  cmdline.isset("condition-coverage-claims") ||
                  cmdline.isset("condition-coverage-rm") ||
                  cmdline.isset("condition-coverage-claims-rm") ||
                  cmdline.isset("branch-coverage") ||
                  cmdline.isset("branch-coverage-claims") ||
                  cmdline.isset("branch-function-coverage") ||
                  cmdline.isset("branch-function-coverage-claims") ||
                  cmdline.isset("k-path-coverage") ||
                  cmdline.isset("k-path-coverage-claims");

    // For coverage mode, treat extra input files (cmdline.args[1:]) as include
    // files so that the coverage location_pool covers all input sources.
    if (is_coverage && cmdline.args.size() > 1)
      for (size_t i = 1; i < cmdline.args.size(); i++)
        config.ansi_c.include_files.push_back(cmdline.args[i]);

    // For Solidity coverage mode: neutralize the multi-transaction harness loop.
    // The _ESBMC_Main_* functions contain a while(nondet_bool()) loop that calls
    // user functions repeatedly. This causes massive symex overhead in coverage
    // mode where we only need each function executed once. Convert backward GOTOs
    // (loop back-edges) in _ESBMC_Main* functions to SKIPs so the loop body
    // executes exactly once.
    if (is_coverage)
    {
      bool is_sol = cmdline.isset("sol");
      if (!is_sol)
        for (const auto &arg : cmdline.args)
          if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".sol")
          {
            is_sol = true;
            break;
          }
      if (is_sol)
      {
        Forall_goto_functions (f_it, goto_functions)
        {
          std::string fname = f_it->first.as_string();
          if (fname.find("_ESBMC_Main") == std::string::npos)
            continue;
          Forall_goto_program_instructions (it, f_it->second.body)
          {
            if (it->is_backwards_goto())
              it->make_skip();
          }
        }
        goto_functions.update();
      }
    }

    // Expand --no-standard-checks before goto_check (also expanded before
    // goto_convert in parse_goto_program; re-expanding here is idempotent
    // and covers the read_goto_binary path).
    if (
      cmdline.isset("no-standard-checks") ||
      options.get_bool_option("no-standard-checks"))
    {
      options.set_option("no-pointer-check", true);
      options.set_option("no-div-by-zero-check", true);
      options.set_option("no-pointer-relation-check", true);
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option("no-vla-size-check", true);
      options.set_option("no-align-check", true);
      options.set_option("no-bounds-check", true);
    }

    // Start by removing all no-op instructions and unreachable code
    if (!(cmdline.isset("no-remove-no-op")))
      remove_no_op(goto_functions);

    // We should skip this 'remove-unreachable' removal in goto-cov and multi-property
    // - multi-property wants to find all the bugs in the src code
    // - assertion-coverage wants to find out unreached codes (asserts)
    // - however, the optimization below will remove codes during the Goto stage
    if (
      !(cmdline.isset("no-remove-unreachable") || is_mul || is_coverage) ||
      cmdline.isset("condition-coverage-rm") ||
      cmdline.isset("condition-coverage-claims-rm"))
      remove_unreachable(goto_functions);

    // Apply all the initialized algorithms
    for (auto &algorithm : goto_preprocess_algorithms)
    {
      if (cmdline.isset("function"))
        algorithm->setTarget(cmdline.getval("function"));
      algorithm->run(goto_functions);
    }

    // Surface dead-store advisories (CWE-563) collected by the pass above.
    // Advisory only: note-level, and it never changes the verification verdict.
    if (!dead_store_advisories.empty())
      for (const auto &adv : dead_store_advisories)
        log_status(
          "{}:{}: {}\n  CWE: {}",
          adv.file,
          adv.line,
          adv.comment,
          format_cwe_list(cwe_for(adv.comment)));

    // Lower throw/catch to symbolic guarded control flow (#5075). Run before
    // inlining so per-call-site exception propagation is still explicit. This
    // is now the only exception path: a program the pass cannot lower is
    // reported as an error rather than silently miscompiled (the legacy
    // imperative path in symex was removed once the lowered subset covered it).
    remove_exceptions(goto_functions, context, ns);

    // do partial inlining
    if (!cmdline.isset("no-inlining"))
    {
      if (cmdline.isset("full-inlining"))
        goto_inline(goto_functions, options, ns);
      else
        goto_partial_inline(goto_functions, options, ns);
    }

    if (cmdline.isset("gcse"))
    {
      std::shared_ptr<value_set_analysist> vsa =
        std::make_shared<value_set_analysist>(ns);
      try
      {
        log_status("Computing Value-Set Analysis (VSA)");
        (*vsa)(goto_functions);
      }
      catch (vsa_not_implemented_exception &)
      {
        log_warning(
          "Unable to compute VSA due to incomplete implementation. Some GOTO "
          "optimizations will be disabled");
        vsa = nullptr;
      }
      catch (type2t::symbolic_type_excp &)
      {
        log_warning(
          "[GOTO] Unable to compute VSA due to symbolic type. Some GOTO "
          "optimizations will be disabled");
        vsa = nullptr;
      }
      catch (const std::string &e)
      {
        log_warning(
          "[GOTO] Unable to compute VSA due to: {}. Some GOTO "
          "optimizations will be disabled",
          e);
        vsa = nullptr;
      }

      if (cmdline.isset("no-library"))
        log_warning("Using CSE with --no-library might cause huge slowdowns!");

      if (!vsa)
        log_warning("Could not apply GCSE optimization due to VSA limitation!");
      else
      {
        goto_cse cse(context, vsa);
        cse.run(goto_functions);
      }
    }

    bool is_k_induction = cmdline.isset("inductive-step") ||
                          cmdline.isset("k-induction") ||
                          cmdline.isset("k-induction-parallel");

    // --termination reuses k-induction's havoc machinery via
    // goto_termination, so the post-havoc invariant injection applies
    // to it too. Treat termination like k-induction for the purpose
    // of interval-analysis pipeline gating.
    bool wants_kind_pipeline =
      is_k_induction || options.get_bool_option("termination");

    if (cmdline.isset("interval-analysis") || cmdline.isset("goto-contractor"))
    {
      // Plain interval analysis without k-induction-style havoc: run
      // with LOOP_MODE so each loop gets a before-back-edge
      // ASSUME(bounds) and an after-loop ASSUME(bounds). The default
      // GUARD_INSTRUCTIONS_LOCAL emits bounds at every assume/assert/
      // goto, which is more uniform but loses the per-loop framing.
      //
      // With k-induction (or termination), stay on
      // GUARD_INSTRUCTIONS_LOCAL: instructions matching k-induction's
      // later transformations rely on the per-instruction bounds being
      // available everywhere, and the post-k-induction pass below adds
      // the at-loop-head bounds that tighten the inductive hypothesis.
      const auto mode =
        wants_kind_pipeline
          ? INTERVAL_INSTRUMENTATION_MODE::GUARD_INSTRUCTIONS_LOCAL
          : INTERVAL_INSTRUMENTATION_MODE::LOOP_MODE;
      interval_analysis(goto_functions, ns, options, mode);
    }

    if (cmdline.isset("validate-correctness-witness"))
    {
      log_status("Enable correctness witness validation 2.0");
      remove_no_op(goto_functions);
      goto_loop_invariant_combined(goto_functions);
    }

    // goto_k_induction returns true when a loop writes an array element
    // through a pointer, which the inductive-step havoc cannot soundly
    // generalise. Disable the inductive step in that case so its UNSAT is
    // not reported as proof (#5224); base case and forward condition run.
    auto disable_is_if_unsound = [&](bool unsound) {
      if (unsound)
      {
        log_warning(
          "k-induction does not support loops that write array elements "
          "through a pointer yet. Disabling inductive step");
        options.set_option("disable-inductive-step", true);
      }
    };

    if (cmdline.isset("loop-invariant"))
    {
      // Combined mode: Branch 1 (invariant inductivity check) +
      // ASSUME(INV) injected at end of loop body + k-induction (Branch 2).
      remove_no_op(goto_functions);
      goto_loop_invariant_combined(goto_functions);
      disable_is_if_unsound(goto_k_induction(goto_functions, ns));
    }
    else
    {
      // --k-induction and --loop-invariant-check are independent and may
      // both be specified.  remove_no_op only needs to run once.
      if (is_k_induction || cmdline.isset("loop-invariant-check"))
        remove_no_op(goto_functions);

      if (is_k_induction)
        disable_is_if_unsound(goto_k_induction(goto_functions, ns));

      if (cmdline.isset("loop-invariant-check"))
      {
        bool use_frame_rule = cmdline.isset("loop-frame-rule");
        goto_loop_invariant(goto_functions, context, use_frame_rule);
      }
    }

    // --termination: reduce non-termination to a reachability safety
    // property by inserting per-loop assert(false) markers and
    // applying the k-induction havoc to every loop. Runs BEFORE
    // instrument_loop_bounds_after_kind so the post-havoc invariant
    // injection sees the transformed IR.
    //
    // Gated on options.get_bool_option, not cmdline.isset: when both
    // --k-induction and --termination are passed, k-induction wins
    // (line ~487 sets options.termination = false). cmdline.isset
    // would still see the original CLI value, incorrectly firing
    // goto_termination on k-induction-only runs.
    if (options.get_bool_option("termination"))
    {
      // Recurrent-set non-termination check (Gupta et al., POPL 2008).
      // Looks for `while(1)`-shaped loops with a constant-equality
      // recurrent set R such that R is reachable from init, closed
      // under some input choice, and disjoint from any exit path. If
      // found, the program is non-terminating and we record it so the
      // verdict loop can report FAILED without unwinding. Never
      // returns TV_TRUE; only TV_FALSE (proved non-terminating) or
      // TV_UNKNOWN.
      bool non_term_proved =
        try_prove_non_termination_by_recurrent_set(goto_functions, options, ns)
          .is_false();
      options.set_option("termination-non-termination-proved", non_term_proved);

      // Ranking-function termination check, on the CLEAN (un-havoced)
      // goto program. If it proves every loop terminates, record it so
      // the verdict loop can report SUCCESSFUL without the marker/FC/IS
      // machinery. Never returns TV_FALSE, so it can only upgrade an
      // UNKNOWN to a proof, never produce a wrong verdict.
      bool ranking_proved =
        try_prove_termination_by_ranking(goto_functions, options, ns).is_true();
      options.set_option("termination-ranking-proved", ranking_proved);

      // Only run the marker/havoc transform when the ranking check did
      // NOT settle it — otherwise the verdict loop short-circuits on the
      // ranking flag and the havoc'd markers would just be dead work.
      if (!ranking_proved)
        goto_termination(goto_functions, options, ns);
    }

    // Pass B (post-k-induction loop bounds): when interval analysis ran
    // earlier and k-induction (or --termination's equivalent havoc) has
    // now finished inserting its nondet havoc before each loop head,
    // recompute bounds with k-induction's preamble instructions treated
    // as transparent, then insert an ASSUME(bounds) right before each
    // loop's exit-test. The ASSUME is marked inductive_step_instruction
    // = true so only the inductive step sees it; base case and forward
    // condition skip it. This is the place where interval analysis
    // actually strengthens the inductive hypothesis.
    if (
      (cmdline.isset("interval-analysis") ||
       cmdline.isset("goto-contractor")) &&
      wants_kind_pipeline)
    {
      instrument_loop_bounds_after_kind(goto_functions, ns, options);
    }

    if (
      cmdline.isset("goto-contractor") ||
      cmdline.isset("goto-contractor-condition"))
    {
#ifdef ENABLE_GOTO_CONTRACTOR
      goto_contractor(goto_functions, ns, options);
#else
      log_error(
        "Current build does not support contractors. If ibex is installed, add "
        "to your build process "
        "-DENABLE_GOTO_CONTRACTOR=ON -DIBEX_DIR=path-to-ibex");
      abort();
#endif
    }

    goto_check(ns, options, goto_functions);

    // Eliminate goto-level no-op loops (empty body, dead modified vars).
    // Runs AFTER goto_check so that any check assertions inserted into a
    // loop body (overflow, div-by-zero, bounds, ...) make body_is_safe
    // refuse the erasure — preserving checks that would otherwise be
    // silently dropped. Skipped under --termination / --unwinding-
    // assertions because loop presence is observable in those modes.
    goto_loop_simplify(goto_functions, options);

    if (options.get_bool_option("atomicity-check"))
      goto_atomicity_check(goto_functions, ns, context);

    // Process function contracts if enabled
    bool has_enforce = cmdline.isset("enforce-contract");
    bool has_replace = cmdline.isset("replace-call-with-contract");
    bool has_enforce_all = cmdline.isset("enforce-all-contracts");
    bool has_replace_all = cmdline.isset("replace-all-contracts");
    if (has_enforce || has_replace || has_enforce_all || has_replace_all)
      process_function_contracts(
        goto_functions,
        has_replace,
        has_enforce,
        has_enforce_all,
        has_replace_all);

    // add re-evaluations of monitored properties
    add_property_monitors(goto_functions, ns);

    // Once again, remove all unreachable and no-op code that could have been
    // introduced by the above algorithms.
    //
    // Skip these cleanups under --termination: goto_termination inserts
    // per-loop ASSERT(false) markers preceded by GOTO orig_target. The
    // GOTO is structurally a "GOTO to next instruction" no-op so
    // remove_no_op would erase it, and ASSERT(false) is treated as
    // having no fall-through successor by get_successors, so
    // remove_unreachable would then strip every original instruction
    // that was only reachable via the marker. Both transformations
    // corrupt the termination CFG. The marker block is intentionally
    // small and ignoring it costs nothing.
    const bool skip_cleanup_for_termination =
      options.get_bool_option("termination");
    if (!(cmdline.isset("no-remove-no-op") || skip_cleanup_for_termination))
      remove_no_op(goto_functions);

    if (!(cmdline.isset("no-remove-unreachable") || is_mul || is_coverage ||
          skip_cleanup_for_termination))
      remove_unreachable(goto_functions);

    goto_functions.update();

    if (
      cmdline.isset("data-races-check") ||
      cmdline.isset("data-races-check-only"))
    {
      log_status("Adding Data Race Checks");
      options.set_option("data-races-check", true);
      options.set_option("no-por", true);
      add_race_assertions(context, goto_functions);
    }

    if (cmdline.isset("restrict-check"))
    {
      log_status("Adding Restrict Aliasing Checks");
      add_restrict_assertions(context, goto_functions);
    }

    //! goto-cov will also mutate the asserts added by esbmc (e.g. goto-check)
    if (
      cmdline.isset("assertion-coverage") ||
      cmdline.isset("assertion-coverage-claims"))
    {
      // for multi-property
      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      options.set_option("no-pointer-check", true);

      // enable '--no-unwinding-assertions' if '--unwind' is enabled
      if (cmdline.isset("unwind"))
        options.set_option("no-unwinding-assertions", true);

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      // for function mode
      if (cmdline.isset("function"))
        tmp.set_target(cmdline.getval("function"));
      tmp.assertion_coverage();
    }

    if (
      cmdline.isset("condition-coverage") ||
      cmdline.isset("condition-coverage-claims") ||
      cmdline.isset("condition-coverage-rm") ||
      cmdline.isset("condition-coverage-claims-rm"))
    {
      // for multi-property
      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      // prevent adding property checking assertions during SymEx
      options.set_option("no-pointer-check", true);
      // unreachable conditions should be also considered as short-circuited

      // enable '--no-unwinding-assertions' if '--unwind' is enabled
      if (cmdline.isset("unwind"))
        options.set_option("no-unwinding-assertions", true);

      // for re-do remove-sideeffects
      options.set_option("goto-instrumented", false);

      //?:
      // if we do not want expressions like 'if(2 || 3)' get simplified to 'if(1||1)'
      // we need to enable the options below:
      //    options.set_option("no-simplify", true);
      //    options.set_option("no-propagation", true);
      // however, this will affect the performance, thus they are not enabled by default

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      // for function mode
      if (cmdline.isset("function"))
        tmp.set_target(cmdline.getval("function"));

      // if we do not want to count the guard in the assertions
      if (cmdline.isset("no-cov-asserts"))
      {
        if (cmdline.isset("cov-assume-asserts"))
          tmp.replace_all_asserts_to_assume();
        else
          tmp.replace_all_asserts_to_guard(gen_true_expr());
      }
      tmp.cov_assume_asserts = cmdline.isset("cov-assume-asserts");
      tmp.condition_coverage();

      // redo conversion to remove_sideeffect
      // Due to that we deliberately skip some of the sideeffects removal process when generating the Goto program.
      // This is to keep the condition/guards format and avoid introducing auxiliary variables, which will affect the coverage calculation.
      goto_coverage_rm temp(context, options, goto_functions);
      temp.remove_sideeffect();
    }

    if (
      cmdline.isset("branch-coverage") ||
      cmdline.isset("branch-coverage-claims"))
    {
      // for multi-property
      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      options.set_option("no-pointer-check", true);

      // enable '--no-unwinding-assertions' if '--unwind' is enabled
      if (cmdline.isset("unwind"))
        options.set_option("no-unwinding-assertions", true);

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      // for function mode
      if (cmdline.isset("function"))
        tmp.set_target(cmdline.getval("function"));
      tmp.cov_assume_asserts = cmdline.isset("cov-assume-asserts");
      tmp.branch_coverage();
    }
    if (
      cmdline.isset("branch-function-coverage") ||
      cmdline.isset("branch-function-coverage-claims"))
    {
      // for multi-property
      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      options.set_option("no-pointer-check", true);

      // enable '--no-unwinding-assertions' if '--unwind' is enabled
      if (cmdline.isset("unwind"))
        options.set_option("no-unwinding-assertions", true);

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      tmp.cov_assume_asserts = cmdline.isset("cov-assume-asserts");
      tmp.branch_function_coverage();
    }

    if (
      cmdline.isset("k-path-coverage") ||
      cmdline.isset("k-path-coverage-claims"))
    {
      // Hard cap on the prefix depth. Goal count per branch grows as
      // 2^(N-1), and (N-1) >= 64 would overflow size_t in `1 << pdepth`.
      // 30 leaves 2^29 goals/branch — already far above any reasonable
      // --k-path-max-goals — and gives a comfortable safety margin from
      // the size_t shift limit. Defense-in-depth: also enforced inside
      // goto_coveraget::k_path_coverage().
      static constexpr int K_PATH_N_MAX = 30;

      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      options.set_option("no-pointer-check", true);
      // Separate boolean enable flag in the option_map. Required because
      // `optionst::get_bool_option(name)` is `atoi(value)`, so storing the
      // CLI int value of `--k-path-coverage` (which is `0` for the no-arg
      // case under boost's implicit_value, or any user-supplied integer)
      // would silently mis-report the feature as disabled in bmc.cpp.
      options.set_option("k-path-coverage-enabled", true);

      if (cmdline.isset("unwind"))
        options.set_option("no-unwinding-assertions", true);

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      if (cmdline.isset("function"))
        tmp.set_target(cmdline.getval("function"));
      tmp.cov_assume_asserts = cmdline.isset("cov-assume-asserts");

      // Resolve N: explicit --k-path-coverage=N > --unwind > fallback 4.
      // The CLI option uses implicit_value(INT_MIN), so `--k-path-coverage`
      // without `=N` parses as INT_MIN (the "no value" sentinel) and falls
      // through to --unwind / 4. Any other non-positive value (incl.
      // explicit `=0` or `=-1`) is rejected — silently falling through
      // would defeat the user's intent.
      const int K_PATH_N_SENTINEL = std::numeric_limits<int>::min();
      int n_arg = K_PATH_N_SENTINEL;
      if (cmdline.isset("k-path-coverage"))
        n_arg = atoi(cmdline.getval("k-path-coverage"));
      if (n_arg > 0)
      {
        if (n_arg > K_PATH_N_MAX)
        {
          log_error(
            "--k-path-coverage=N requires 1 <= N <= {} (got {})",
            K_PATH_N_MAX,
            n_arg);
          return true;
        }
        tmp.k_path_n = static_cast<size_t>(n_arg);
      }
      else if (n_arg != K_PATH_N_SENTINEL)
      {
        // Explicit non-positive value — reject rather than silently
        // falling back.
        log_error(
          "--k-path-coverage=N requires 1 <= N <= {} (got {})",
          K_PATH_N_MAX,
          n_arg);
        return true;
      }
      else if (cmdline.isset("unwind"))
      {
        int u = atoi(cmdline.getval("unwind"));
        if (u <= 0 || u > K_PATH_N_MAX)
        {
          log_error(
            "--k-path-coverage cannot derive N from --unwind={} (must be "
            "in 1..{}); pass --k-path-coverage=N explicitly",
            u,
            K_PATH_N_MAX);
          return true;
        }
        tmp.k_path_n = static_cast<size_t>(u);
      }
      else
      {
        tmp.k_path_n = 4;
        log_status(
          "--k-path-coverage: no N or --unwind specified; defaulting to "
          "N=4");
      }

      auto read_positive = [&](const char *flag, size_t &dst) -> bool {
        if (!cmdline.isset(flag))
          return true;
        int v = atoi(cmdline.getval(flag));
        if (v <= 0)
        {
          log_error("--{} requires a positive integer (got {})", flag, v);
          return false;
        }
        dst = static_cast<size_t>(v);
        return true;
      };
      if (!read_positive("k-path-witness-depth", tmp.k_path_witness_depth))
        return true;
      if (!read_positive("k-path-max-goals", tmp.k_path_max_goals))
        return true;

      tmp.k_path_coverage();
    }

    if (cmdline.isset("negating-property"))
    {
      std::string tgt_fname = cmdline.getval("negating-property");
      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      tmp.negating_asserts(tgt_fname);
    }
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}
