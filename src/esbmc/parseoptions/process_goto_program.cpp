#include <ac_config.h>

#include <esbmc/bmc.h>
#include <esbmc/esbmc_parseoptions.h>
#include <util/config/focus_function.h>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/trace/sarif.h>
#include <util/base/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config/config.h>
#include <util/base/filesystem.h>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <util/arith/ieee_float.h>
#include <util/expr/expr_util.h>
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
#include <goto-programs/goto_invariant_synthesis.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/gcse.h>
#include <goto-programs/loop_numbers.h>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/write_goto_binary.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_library_assertions.h>
#include <goto-programs/remove_exceptions.h>
#include <goto-programs/lower_longjmp.h>
#include <goto-programs/set_claims.h>
#include <goto-programs/show_claims.h>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/goto_check_uninit_vars.h>
#include <goto-programs/goto_check_unchecked_return.h>
#include <goto-programs/dead_store_analysis.h>
#include <goto-programs/mark_decl_as_non_det.h>
#include <goto-programs/assign_params_as_non_det.h>
#include <goto2c/goto2c.h>
#include <util/irep/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/andersen.h>
#include <util/symtab/symbol.h>
#include <util/base/time_stopping.h>
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

/* Rewrite the initialiser __ESBMC_main performs for a model-tuning global.
 * Rewriting the initialiser rather than the symbol leaves any later assignment
 * (fesetround(), say) free to change the value again. Returns false when the
 * program does not carry the global, which happens when nothing pulled the
 * model in. */
static bool set_model_global(
  goto_functionst &goto_functions,
  const irep_idt &name,
  const BigInt &value)
{
  goto_functionst::function_mapt::iterator main =
    goto_functions.function_map.find("__ESBMC_main");
  if (main == goto_functions.function_map.end())
    return false;

  for (auto &i : main->second.body.instructions)
  {
    if (!i.is_assign())
      continue;
    const code_assign2t &assign = to_code_assign2t(i.code);
    if (
      !is_symbol2t(assign.target) || to_symbol2t(assign.target).thename != name)
      continue;
    i.code = code_assign2tc(
      assign.target, constant_int2tc(assign.source->type, value));
    return true;
  }
  return false;
}

/* --fp-taylor-terms tunes how far the exp/log/pow models expand their Taylor
 * series (esbmc/esbmc#2865); see src/c2goto/library/libm/exp.c. */
static bool
apply_taylor_terms(goto_functionst &goto_functions, const cmdlinet &cmdline)
{
  if (!cmdline.isset("fp-taylor-terms"))
    return false;

  const int terms = atoi(cmdline.getval("fp-taylor-terms"));
  if (terms < 2 || terms > 12)
  {
    log_error("--fp-taylor-terms must be between 2 and 12; got {}", terms);
    return true;
  }

  if (!set_model_global(
        goto_functions, "c:@__ESBMC_fp_taylor_terms", BigInt(terms)))
    log_warning(
      "--fp-taylor-terms has no effect: this program calls no Taylor-series "
      "model");
  return false;
}

/* The --round-to-* flags select the IEEE 754 rounding mode (esbmc/esbmc#2763).
 * Floating-point operations read it from __ESBMC_rounding_mode, a global whose
 * zero-initialiser __ESBMC_main performs. */
static bool
apply_rounding_mode(goto_functionst &goto_functions, const cmdlinet &cmdline)
{
  static const struct
  {
    const char *flag;
    ieee_floatt::rounding_modet mode;
  } modes[] = {
    {"round-to-nearest", ieee_floatt::ROUND_TO_EVEN},
    {"round-to-even", ieee_floatt::ROUND_TO_EVEN},
    {"round-to-plus-inf", ieee_floatt::ROUND_TO_PLUS_INF},
    {"round-to-minus-inf", ieee_floatt::ROUND_TO_MINUS_INF},
    {"round-to-zero", ieee_floatt::ROUND_TO_ZERO}};

  const char *selected_flag = nullptr;
  ieee_floatt::rounding_modet selected = ieee_floatt::ROUND_TO_EVEN;
  for (const auto &m : modes)
  {
    if (!cmdline.isset(m.flag))
      continue;
    if (selected_flag && selected != m.mode)
    {
      log_error("--{} and --{} are mutually exclusive", selected_flag, m.flag);
      return true;
    }
    selected_flag = m.flag;
    selected = m.mode;
  }

  if (!selected_flag)
    return false;

  if (!set_model_global(
        goto_functions, "c:@__ESBMC_rounding_mode", BigInt(selected)))
    log_warning(
      "--{} has no effect: this program performs no floating-point rounding",
      selected_flag);
  return false;
}

static void apply_gcse(
  goto_functionst &goto_functions,
  contextt &context,
  const cmdlinet &cmdline)
{
  // The available-expressions analysis is sequential: another thread may
  // write between two reads it treats as equal (#8007).
  if (spawns_threads(goto_functions))
  {
    log_warning("--gcse is ignored: the program may create threads");
    return;
  }

  auto andersen = std::make_shared<andersent>();
  log_status("Computing points-to analysis (Andersen)");
  (*andersen)(goto_functions);
  std::shared_ptr<value_setst> points_to = andersen;

  if (cmdline.isset("no-library"))
    log_warning("Using CSE with --no-library might cause huge slowdowns!");

  goto_cse cse(context, points_to);
  cse.run(goto_functions);
}

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
// Exceptions whose escape is not reported as a failed property.
static std::set<irep_idt> unreported_exceptions(const optionst &options)
{
  if (options.get_bool_option("python-ignore-assertions"))
    return {"AssertionError"};
  return {};
}

bool esbmc_parseoptionst::process_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    namespacet ns(context);

    if (
      apply_rounding_mode(goto_functions, cmdline) ||
      apply_taylor_terms(goto_functions, cmdline))
      return true;

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
                  cmdline.isset("k-path-coverage-claims") ||
                  cmdline.isset("solidity-path-coverage") ||
                  cmdline.isset("dead-code-check");

    // Single source of truth for "this run measures coverage", read back by
    // bmc.cpp. --dead-code-check borrows the same instrumentation but keeps a
    // verdict and reports CWE-561 advisories, so it is deliberately excluded:
    // the coverage reporting rules must not apply to it (issue #6387).
    options.set_option(
      "coverage-measurement", is_coverage && !cmdline.isset("dead-code-check"));

    // Claims after a violation are still checked, so no pass may treat a
    // failed assertion as the end of its path (#7900).
    const bool checks_past_violation =
      options.get_bool_option("multi-property") || is_coverage;

    // For coverage mode, treat extra input files (cmdline.args[1:]) as include
    // files so that the coverage location_pool covers all input sources.
    if (is_coverage && cmdline.args.size() > 1)
      for (size_t i = 1; i < cmdline.args.size(); i++)
        config.ansi_c.include_files.push_back(cmdline.args[i]);

    // For Solidity coverage mode: neutralize the multi-transaction harness
    // loop. The _ESBMC_Main_* functions contain a while(nondet_bool()) loop
    // that calls user functions repeatedly. This causes massive symex overhead
    // in coverage mode where we only need each function executed once. Convert
    // backward GOTOs (loop back-edges) in _ESBMC_Main* functions to SKIPs so
    // the loop body executes exactly once.
    //
    // --coverage-multi-tx opts out: the loop stays live (bounded by --unwind),
    // so a branch reachable only through a state-building call sequence
    // (deposit(); withdraw();) becomes coverable and the Foundry generator
    // can reconstruct the multi-call sequence.
    if (is_coverage && !cmdline.isset("coverage-multi-tx"))
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

    // Re-applied here for the read_goto_binary path, which bypasses
    // parse_goto_program.
    set_solidity_check_defaults(options);

    // Expand --no-standard-checks before goto_check (also expanded before
    // goto_convert in parse_goto_program; re-expanding here is idempotent
    // and covers the read_goto_binary path).
    if (
      cmdline.isset("no-standard-checks") ||
      options.get_bool_option("no-standard-checks"))
    {
      auto set_neg_unless_pos = [&](const char *neg, const char *pos) {
        if (!cmdline.isset(pos))
          options.set_option(neg, true);
      };
      // See the goto_convert-side block above: --bounds-check re-enables the
      // pointer-deref check so symbolic-size dynamic-array OOB (a dereference
      // failure) is detectable. A contradictory `--bounds-check
      // --no-bounds-check` keeps pointer-check disabled too.
      if (!cmdline.isset("bounds-check") || cmdline.isset("no-bounds-check"))
        options.set_option("no-pointer-check", true);
      set_neg_unless_pos("no-div-by-zero-check", "div-by-zero-check");
      options.set_option("no-pointer-relation-check", true);
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option("no-vla-size-check", true);
      options.set_option("no-align-check", true);
      set_neg_unless_pos("no-bounds-check", "bounds-check");
      set_neg_unless_pos("no-narrowing-check", "narrowing-check");
    }

    // Must precede inlining: goto_inlinet relabels a hidden function's
    // instructions with the call site, after which the models' assertions
    // can no longer be told apart from the user's (discussion #6382).
    if (options.get_bool_option("no-library-assertions"))
      remove_library_assertions(goto_functions);

    // Start by removing all no-op instructions and unreachable code
    if (!(cmdline.isset("no-remove-no-op")))
      remove_no_op(goto_functions);

    // We should skip this 'remove-unreachable' removal in goto-cov and
    // multi-property
    // - multi-property wants to find all the bugs in the src code
    // - assertion-coverage wants to find out unreached codes (asserts)
    // - however, the optimization below will remove codes during the Goto stage
    if (
      !(cmdline.isset("no-remove-unreachable") || checks_past_violation) ||
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

    lower_longjmp(goto_functions, context, options);

    // Lower throw/catch to symbolic guarded control flow (#5075). Run before
    // inlining so per-call-site exception propagation is still explicit. This
    // is now the only exception path: a program the pass cannot lower is
    // reported as an error rather than silently miscompiled (the legacy
    // imperative path in symex was removed once the lowered subset covered it).
    remove_exceptions(
      goto_functions,
      context,
      ns,
      unreported_exceptions(options),
      python_checked_exceptions(options));

    // do partial inlining
    if (!cmdline.isset("no-inlining"))
    {
      if (cmdline.isset("full-inlining"))
        goto_inline(goto_functions, options, ns);
      else
        goto_partial_inline(goto_functions, options, ns);
    }

    if (cmdline.isset("gcse"))
      apply_gcse(goto_functions, context, cmdline);

    // Under --termination, goto_termination does its own havoc, so
    // goto_k_induction must not also run (#6031).
    bool is_k_induction =
      (cmdline.isset("inductive-step") || cmdline.isset("k-induction") ||
       cmdline.isset("k-induction-parallel") || cmdline.isset("ts-check")) &&
      !options.get_bool_option("termination");

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
      interval_analysis(
        goto_functions, ns, options, checks_past_violation, mode);
    }

    if (cmdline.isset("validate-correctness-witness"))
    {
      log_status("Enable correctness witness validation 2.0");
      remove_no_op(goto_functions);
      goto_loop_invariant_combined(goto_functions);
    }

    // goto_k_induction returns true, and logs why, when the inductive step
    // would be unsound: e.g. a loop writes through a pointer to storage its
    // havoc cannot name (#5224). Disable the step then so its UNSAT is not
    // reported as proof; base case and forward condition run.
    auto disable_is_if_unsound = [&](bool unsound) {
      if (unsound)
        options.set_option("disable-inductive-step", true);
    };

    if (cmdline.isset("loop-invariant"))
    {
      // Combined mode: Branch 1 (invariant inductivity check) +
      // ASSUME(INV) injected at end of loop body + k-induction (Branch 2).
      remove_no_op(goto_functions);
      goto_loop_invariant_combined(goto_functions);
      disable_is_if_unsound(
        goto_k_induction(goto_functions, ns, checks_past_violation));
    }
    else
    {
      // --k-induction and --loop-invariant-check are independent and may
      // both be specified.  remove_no_op only needs to run once.
      if (is_k_induction || wants_loop_invariants())
        remove_no_op(goto_functions);

      if (is_k_induction)
        disable_is_if_unsound(
          goto_k_induction(goto_functions, ns, checks_past_violation));

      if (wants_loop_invariants())
        apply_loop_invariants(goto_functions, context, options, is_k_induction);
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
      instrument_loop_bounds_after_kind(
        goto_functions, ns, options, checks_past_violation);
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
    {
      if (
        process_function_contracts(
          goto_functions,
          has_replace,
          has_enforce,
          has_enforce_all,
          has_replace_all))
        return true;
    }

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

    if (!(cmdline.isset("no-remove-unreachable") || checks_past_violation ||
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

    if (cmdline.isset("restrict-assume"))
    {
      log_status("Assuming Restrict Parameters Do Not Alias");
      add_restrict_assumptions(
        context, goto_functions, config.main.empty() ? "main" : config.main);
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
      // if we do not want expressions like 'if(2 || 3)' get simplified to
      // 'if(1||1)' we need to enable the options below:
      //    options.set_option("no-simplify", true);
      //    options.set_option("no-propagation", true);
      // however, this will affect the performance, thus they are not enabled by
      // default

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
      // Due to that we deliberately skip some of the sideeffects removal
      // process when generating the Goto program. This is to keep the
      // condition/guards format and avoid introducing auxiliary variables,
      // which will affect the coverage calculation.
      goto_coverage_rm temp(context, options, goto_functions);
      temp.remove_sideeffect();
    }

    // --dead-code-check reuses the branch-coverage instrumentation: each
    // conditional branch gets `assert(guard)` and `assert(!guard)` reachability
    // probes, and an unviolated probe means the *opposite* direction is dead.
    // report_dead_code() reports those as CWE-561 note-level advisories without
    // flipping the verdict (see bmc.cpp).
    if (
      cmdline.isset("branch-coverage") ||
      cmdline.isset("branch-coverage-claims") ||
      cmdline.isset("dead-code-check"))
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
      // Per-contract branch coverage (semantics A): scope the
      // denominator/numerator to decisions lexically declared inside the
      // --contract target only. --coverage-whole-unit opts out: keep the
      // contract as the harness entry but count the whole compilation unit.
      if (cmdline.isset("contract") && !cmdline.isset("coverage-whole-unit"))
        tmp.scope_contract = cmdline.getval("contract");
      // Cross-run persisted covered-set: edges already witnessed in a
      // prior run are not re-instrumented (the denominator is still the
      // full static universe, so % is never inflated).
      if (cmdline.isset("coverage-covered-set"))
        tmp.covered_set_path = cmdline.getval("coverage-covered-set");
      // Dependency exclusion (Item 5-d): decisions whose declaring
      // contract is in this set are dropped from BOTH denominator and
      // numerator. Meaningful under --coverage-whole-unit (a no-op in
      // default mode, where scope_contract already filters foreign code).
      if (cmdline.isset("coverage-exclude-contract"))
        for (const auto &c : cmdline.get_values("coverage-exclude-contract"))
          tmp.exclude_contracts.insert(c);
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

    if (cmdline.isset("solidity-path-coverage"))
    {
      // `--multi-fail-fast N` stops after N satisfiable claims and abandons the
      // rest. Every abandoned path claim then has NO verdict, and the report
      // cannot tell "not reachable" from "never asked" — it would silently
      // report those paths as undecided while the run looked successful. The
      // whole point of the tri-state is that such a difference is visible, so
      // reject the combination instead of quietly producing a truncated ledger.
      if (cmdline.isset("multi-fail-fast"))
      {
        log_error(
          "--solidity-path-coverage is incompatible with --multi-fail-fast: "
          "fail-fast abandons the remaining path claims, so the report could "
          "not distinguish a path that is unreachable from one that was never "
          "solved. Drop --multi-fail-fast (every path must get a verdict).");
        return true;
      }

      // Mirror the branch/k-path dispatch: coverage runs as base-case
      // multi-property so each instrumented path assert is checked
      // independently (a violated assert == that complete path is feasible).
      options.set_option("base-case", true);
      options.set_option("multi-property", true);
      options.set_option("keep-verified-claims", false);
      options.set_option("no-pointer-check", true);
      // Dedicated boolean enable flag: get_bool_option is atoi(), and a NULL
      // flag stores "" (atoi -> 0), so bmc.cpp keys is_goto_cov off this.
      options.set_option("solidity-path-coverage-enabled", true);

      std::string filename = cmdline.args[0];
      goto_coveraget tmp(ns, goto_functions, filename);
      // Ghost snapshot symbols are moved into `context` (available here).
      tmp.cov_context = &context;
      if (cmdline.isset("function"))
        tmp.set_target(cmdline.getval("function"));
      // --contract scoping: only enumerate functions declared in the target
      // contract (not sibling contracts). --coverage-whole-unit opts out.
      if (cmdline.isset("contract") && !cmdline.isset("coverage-whole-unit"))
        tmp.scope_contract = cmdline.getval("contract");
      // --focus-function scoping. The frontend already narrows which entry the
      // DISPATCHER may call; this narrows what gets ENUMERATED AND
      // INSTRUMENTED, so a focused run's published numbers describe the focused
      // unit instead of the whole contract. Measured on aqua `--focus-function
      // dock`: 2846 paths in the denominator, of which 2783 belong to units the
      // dispatcher cannot enter in that run, so `Path Coverage` read 0.07%
      // where the honest figure against `dock`'s own 63 paths is 3.17%.
      //
      // Read here rather than inside the pass, like every other knob, so the
      // pass keeps having no command-line dependency of its own.
      if (cmdline.isset("focus-function"))
        tmp.focus_function = cmdline.getval("focus-function");
      // ---- --path-cov-instrument-only: DISPATCH WIDE, INSTRUMENT NARROW ----
      //
      // The option above answers TWO questions with one value -- which entries
      // the harness may CALL, and which units are MEASURED -- and a
      // state-guarded path needs opposite answers to them: a second function in
      // the dispatcher so the state can be established, and a denominator that
      // does not move so the ladder's cells stay comparable. MEASURED on aqua:
      // `--focus-function dock,ship` raised the instrumented set from dock's 63
      // paths to 2796 (ship contributes 2733) and the run was killed at the
      // 300 s outer timeout with no usable answer, at tx=1 and again at tx=2.
      //
      // THE SUBSET IS CHECKED, NOT ASSUMED. `util/config/focus_function.h` is
      // right that the dispatcher filter and the instrumentation filter must
      // not disagree; the direction it names -- dispatchable but uninstrumented
      // -- is what this option deliberately creates and it is the harmless one
      // (a smaller, honestly-published denominator). The other direction is
      // not: an instrumented unit the dispatcher cannot enter reports every
      // path `unit-not-entered`, which reads as "nothing reaches this code" and
      // means "nothing was asked to". So every name here must also be selected
      // by --focus-function, and a name that is not REFUSES the run.
      //
      // An EMPTY --focus-function selects everything (focus_function.h), so
      // `--path-cov-instrument-only dock` with no focus at all is legal and is
      // the widest useful cell: the whole contract in the dispatcher, one
      // unit's paths in the denominator.
      if (cmdline.isset("path-cov-instrument-only"))
      {
        const std::string io = cmdline.getval("path-cov-instrument-only");
        const std::vector<std::string> io_names = focus_function_names(io);
        if (io_names.empty())
        {
          log_error(
            "--path-cov-instrument-only was given the value '{}', which names "
            "no function at all. Pass one or more unit names, separated by "
            "commas or spaces.",
            io);
          return true;
        }
        std::string outside;
        for (const auto &n : io_names)
          if (!focus_function_selects(tmp.focus_function, n))
            outside += (outside.empty() ? "" : ", ") + ("'" + n + "'");
        if (!outside.empty())
        {
          log_error(
            "--path-cov-instrument-only names {}, which --focus-function '{}' "
            "does not select. An instrumented unit the dispatcher cannot enter "
            "reports every one of its paths as 'unit-not-entered' -- a zero "
            "that reads as 'nothing reaches this code' when it means 'nothing "
            "was asked to'. Add the name to --focus-function, or drop "
            "--focus-function entirely (an empty focus dispatches everything).",
            outside,
            tmp.focus_function);
          return true;
        }
        tmp.instrument_only = io;
      }
      // Cross-run persisted covered-set: a complete path already witnessed
      // (CE in hand) in an earlier escalation round is not re-instrumented,
      // so each round spends its budget only on paths still lacking a CE.
      // The denominator stays the full enumerated path set, so the reported
      // coverage is never inflated by the skip.
      if (cmdline.isset("coverage-covered-set"))
        tmp.covered_set_path = cmdline.getval("coverage-covered-set");
      // The JSON report's reason for existing is the per-path counterexample
      // payload (call arguments / EVM environment / post-state). A path claim's
      // guard mentions nothing but the ghost accumulators, so the symex slicer
      // — which keeps only what the claim depends on — removes every state
      // write and every environment read, and the payload comes back EMPTY.
      // Exempt exactly the symbols the harvest reads (see protect_ce_symbols)
      // instead of switching slicing off wholesale: the latter also keeps every
      // c2goto crypto/ABI table in the formula, for no benefit to the report.
      if (cmdline.isset("cov-report-json"))
      {
        tmp.protect_ce_symbols = true;
        // ...and the payload has to survive a run that does not finish. The
        // report is written once, after the job loop and inside the try an OOM
        // unwinds, so a run that dies keeps none of it. The journal is written
        // at the moment each path is witnessed. See path_ce_journal_path.
        tmp.emit_ce_journal = true;
        // The report is also the only place the per-path DECISION SEQUENCE can
        // be published, and it is what puts path coverage and branch coverage
        // on one denominator (the decisions walked). Recorded only for a run
        // that asks for the report, because the memory is per path PREFIX.
        tmp.emit_decision_sites = true;
      }
      // Per-unit path budget. Read BEFORE solidity_path_coverage() because the
      // pass uses it twice and in a fixed order: first as the target that
      // degradation withdraws call points to reach, then as the goal cap that
      // truncates whatever degradation could not fit. It also enters the
      // cross-run fingerprint, so changing it discards the covered set rather
      // than silently reusing entries computed under a different budget.
      if (cmdline.isset("path-cov-max-goals"))
      {
        const int v = atoi(cmdline.getval("path-cov-max-goals"));
        if (v <= 0)
        {
          log_error(
            "--path-cov-max-goals requires a positive integer (got {})", v);
          return true;
        }
        tmp.path_cov_max_goals = static_cast<size_t>(v);
      }
      // ---- THE PER-CLAIM SOLVER BUDGET ----
      //
      // Read here with the other knobs, and published as a static so the solve
      // loop and the report read the same number rather than each re-deriving
      // it. Default 120 s, 0 = unlimited. Negative is refused rather than
      // clamped: a negative budget has no reading that is not a mistake, and
      // silently turning it into "unlimited" would give the caller the opposite
      // of what they typed.
      {
        int t = 120;
        if (cmdline.isset("path-cov-claim-timeout"))
          t = atoi(cmdline.getval("path-cov-claim-timeout"));
        if (t < 0)
        {
          log_error(
            "--path-cov-claim-timeout requires a non-negative integer (got "
            "{}); use 0 for unlimited",
            t);
          return true;
        }
        goto_coveraget::claim_budget_seconds = static_cast<size_t>(t);
        goto_coveraget::claim_budget_exceeded.store(
          0, std::memory_order_relaxed);
        goto_coveraget::claim_budget_mechanism.clear();
        // Published as MILLISECONDS into `options` so the solver backends can
        // read it. It has to be republished rather than read from the CLI
        // there: boost never pumps a DEFAULTED value into `optionst`
        // (optionst::cmdline, util/options.cpp), so an untouched
        // --path-cov-claim-timeout would read as empty in the backend and the
        // 120 s default would silently never apply -- a budget that is not
        // applied while the report says it was is worse than no budget.
        options.set_option(
          "path-cov-claim-timeout-ms", std::to_string((long long)t * 1000));
      }
      // Stage-2 certification query. Read here, alongside the other knobs the
      // pass consumes, so the pass itself has no command-line dependency.
      if (cmdline.isset("path-cov-certify"))
        tmp.path_cov_certify_path = cmdline.getval("path-cov-certify");
      if (cmdline.isset("path-cov-outer-box"))
        tmp.path_cov_outer_box_path = cmdline.getval("path-cov-outer-box");
      // Stage-3 post-state assertion synthesis. Read beside the other two so
      // the pass keeps having no command-line dependency of its own.
      if (cmdline.isset("path-cov-assert"))
      {
        tmp.path_cov_assert_path = cmdline.getval("path-cov-assert");
        // FORCED, for the same reason --unwind 4 is forced above: a candidate
        // that is TRUE gets discharged during simplification and never enters
        // the verdict ledger, which only records claims the solve loop filed.
        // The reporter then has nothing to read for it and prints "NO VERDICT
        // (never reached the solver)" -- turning the mode's WANTED outcome into
        // a non-answer, silently.
        //
        // MEASURED on the R1 must-flip pair. Without it:
        //   0 HOLDS, 3 REFUTED, 3 no verdict (never reached the solver)
        // With it:
        //   3 HOLDS, 3 REFUTED, 0 no verdict
        // Same program, same region, same six candidates. A mode whose entire
        // output is a HOLDS / REFUTED table cannot let HOLDS become no-verdict
        // as a side effect of an optimisation.
        options.set_option("no-simplify", true);
      }

      // ---- THE THREE STAGE-2/3 MODES ARE MUTUALLY EXCLUSIVE ----
      //
      // They are branches at the end of solidity_path_coverage() and each one
      // `continue`s out of the per-unit loop, so the FIRST one tested wins and
      // the others never fire. That precedence is silent and it already has a
      // measured consequence: `--path-cov-outer-box` together with
      // `--path-cov-certify` runs the outer-box branch, certify emits not one
      // assume and not one assert, `certify_units_matched` stays 0, and the run
      // then dies at the route-5 gate with a message blaming the UNIT NAME --
      // pointing the reader at a spelling mistake that does not exist.
      //
      // Rejected here rather than ordered here, because "which one wins" is not
      // a question with a right answer: the three modes ask three different
      // questions and a caller that passed two of them does not know which one
      // it got. Adding a third branch without this gate would have turned one
      // silent precedence into three.
      {
        std::vector<std::string> stage2;
        if (cmdline.isset("path-cov-outer-box"))
          stage2.push_back("--path-cov-outer-box");
        if (cmdline.isset("path-cov-certify"))
          stage2.push_back("--path-cov-certify");
        if (cmdline.isset("path-cov-assert"))
          stage2.push_back("--path-cov-assert");
        if (stage2.size() > 1)
        {
          std::string names;
          for (const auto &n : stage2)
            names += (names.empty() ? "" : ", ") + n;
          log_error(
            "--solidity-path-coverage: {} were given together ({}). These are "
            "three mutually exclusive stage-2/3 modes implemented as three "
            "branches at the end of one pass, and each one leaves the unit "
            "loop "
            "as soon as it fires -- so passing two does not run two, it runs "
            "the first and silently discards the rest. Historically that "
            "discarded run then failed with a message about the unit NAME, "
            "which is not what was wrong. Pass exactly one.",
            stage2.size(),
            names);
          return true;
        }
      }
      tmp.cov_assume_asserts = cmdline.isset("cov-assume-asserts");
      // Align the offline enumeration bound with the symex unwind bound. The
      // two MUST agree, or "this path is feasible" as enumerated and "this path
      // is feasible" as explored are answers to different questions.
      //
      // With --unwind unset they did not agree: the enumeration bounded every
      // back-edge at path_cov_unwind (4) while symex was unbounded. That is not
      // just a mismatch on paper — it is fatal in practice, because a Solidity
      // external call is modelled as a nondet RE-ENTRY into the contract's own
      // dispatcher, so `t.call("")` recurses without a bound. Measured: a
      // two-function contract with one `.call` unwound
      // `_ESBMC_Nondet_Extcall_C` 944 times and died with `ERROR: Out of
      // memory`; a plain `for` loop with no --unwind did the same. Both
      // terminate immediately once a bound exists.
      //
      // So: honour an explicit --unwind, and otherwise ADOPT the enumeration's
      // own bound as the unwind bound and say so. Truncation stays visible —
      // report_coverage prints the truncated-loop warning and every JSON entry
      // carries bound.unwind — so this buys termination without hiding what was
      // cut.
      if (cmdline.isset("unwind"))
      {
        int u = atoi(cmdline.getval("unwind"));
        if (u > 0)
          tmp.path_cov_unwind = static_cast<size_t>(u);
      }
      else
      {
        options.set_option("unwind", std::to_string(tmp.path_cov_unwind));
        log_status(
          "--solidity-path-coverage: no --unwind given; bounding symbolic "
          "execution at {} to match the path enumeration's own loop bound. "
          "Without it an external call (modelled as nondet re-entry into this "
          "contract's dispatcher) or any loop runs unbounded until the memory "
          "limit. Pass --unwind N to choose a different bound",
          tmp.path_cov_unwind);
      }
      options.set_option("no-unwinding-assertions", true);
      tmp.solidity_path_coverage();
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

/// Whether the run needs the loop-invariant machinery.
/// --synthesise-loop-invariants supplies the invariants that
/// --loop-invariant-check discharges, so it implies that mode.
bool esbmc_parseoptionst::wants_loop_invariants() const
{
  return cmdline.isset("loop-invariant-check") ||
         cmdline.isset("synthesise-loop-invariants");
}

/// Synthesise the invariants when asked, then run the schema that discharges
/// them.
void esbmc_parseoptionst::apply_loop_invariants(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  bool k_induction_ran)
{
  if (cmdline.isset("synthesise-loop-invariants"))
    // Read from `options`, the same object goto_check consults
    // (goto_check.cpp:34,36). Deciding the same question from `cmdline`
    // instead happens to agree today only because nothing calls
    // set_option on these two, which is a property of the current code
    // rather than an enforced one.
    goto_synthesise_loop_invariants(
      goto_functions,
      overflow_checkst{
        options.get_bool_option("overflow-check"),
        options.get_bool_option("unsigned-overflow-check")},
      k_induction_ran);

  bool use_frame_rule = cmdline.isset("loop-frame-rule");
  goto_loop_invariant(goto_functions, context, use_frame_rule);
}
