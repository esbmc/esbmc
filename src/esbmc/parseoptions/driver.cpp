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
#ifdef ENABLE_SOLIDITY_FRONTEND
#  include <solidity-frontend/solidity_tod_analysis.h>
#  include <solidity-frontend/solidity_tod_harness.h>
#endif
// The one parser for --focus-function's value; --path-cov-instrument-only is
// validated as a subset of it with the same matcher the frontend dispatcher
// and the path-coverage pass use.
#include <util/config/focus_function.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/trace/sarif.h>
#include <util/base/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <irep2/simplification_check.h>
#include <solvers/smt/simplification_equivalence.h>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config/config.h>
#include <util/base/filesystem.h>
#include <csignal>
#include <cstdlib>
#include <limits>
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
#include <goto-programs/goto_check_excessive_alloc.h>
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
#include <pointer-analysis/value_set_analysis.h>
#include <util/symtab/symbol.h>
#include <util/base/time_stopping.h>
#include <goto-programs/goto_cfg.h>
#include <langapi/language_util.h>
#include <goto-programs/contracts/contracts.h>
#include <util/ssa/proof_cache.h>

#ifdef _WIN32
// MSVC spells the pipe-capable subprocess helpers with a leading underscore.
#  define popen _popen
#  define pclose _pclose
#endif

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

// ANSI color/style escape sequences for terminal output
#define CLR_BOLD_CYAN "\033[1;36m"
#define CLR_BOLD "\033[1m"
#define CLR_RESET "\033[0m"

/// The forked k-induction processes each keep their own property table, and
/// the one that concludes the run reports its base-case bound as PASSED (D4 of
/// docs/roadmap/multi-property-strategy-plan.md). The other flags listed turn
/// --multi-property on. --termination runs the sequential driver instead.
static bool k_induction_parallel_conflicts(const cmdlinet &cmdline)
{
  if (!cmdline.isset("k-induction-parallel") || cmdline.isset("termination"))
    return false;

  for (const char *incompatible :
       {"multi-property", "all-witnesses", "parallel-solving"})
    if (cmdline.isset(incompatible))
    {
      log_error(
        "--k-induction-parallel cannot be combined with --{}", incompatible);
      return true;
    }
  return false;
}

/// The flag combinations that cannot produce a sound report, rejected
/// before any work is done. Kept out of doit() so every combination check
/// reads in one place.
static bool incompatible_flags(const cmdlinet &cmdline)
{
  // --dead-code-check is a standalone base-case advisory analysis: it reuses
  // the branch-coverage instrumentation and forces a SUCCESSFUL verdict. The
  // k-induction / incremental strategies and early-stopping fail-fast drive
  // paths it neither exercises nor can report soundly (a stopped fail-fast
  // run leaves unsolved probes that would be misreported as dead), so reject
  // those combinations up front (issue #4495).
  if (cmdline.isset("dead-code-check"))
    for (const char *incompatible :
         {"k-induction",
          "k-induction-parallel",
          "incremental-bmc",
          "falsification",
          "termination",
          "loop-invariant",
          "multi-fail-fast",
          // Standalone phase-only modes: the dead-code report is only emitted
          // on the base-case pass (bmc.cpp `bs && !fc && !is`), so a run under
          // --forward-condition / --inductive-step would exit SUCCESSFUL
          // without ever printing the [Dead code] findings.
          "forward-condition",
          "inductive-step",
          // Claim filtering marks every unselected assertion SKIP but leaves it
          // in goto_coveraget::all_claims, so its probe never solves and the
          // reporter would flag live branches as dead. SMT-formula emission
          // (P_SMTLIB) likewise leaves every probe unsolved. --no-assertions
          // drops the probes outright: goto_coveraget::insert_assert marks them
          // user_provided(true), which is just what symex_assert discards, so
          // every branch would be reported dead. All produce spurious CWE-561
          // findings, so reject them (issue #4495 / #5934).
          "claim",
          "smt-formula-only",
          "smt-formula-too",
          "no-assertions",
          // Other coverage modes reuse the same goto_coveraget::all_claims and
          // multi_property_check routing: assertion coverage diverts solved
          // probes into reached_mul_claims (leaving reached_claims empty), and
          // the branch-function / k-path passes overwrite all_claims after the
          // dead-code instrumentation runs. Either way the reporter would mark
          // live branches as dead, so reject them (issue #4495).
          "assertion-coverage",
          "assertion-coverage-claims",
          "condition-coverage",
          "condition-coverage-claims",
          "condition-coverage-rm",
          "condition-coverage-claims-rm",
          "branch-function-coverage",
          "branch-function-coverage-claims",
          "k-path-coverage",
          "k-path-coverage-claims",
          // Safety checks injected during symex (after the coverage
          // instrumentation) surface as genuine SAT claims that are not in
          // all_claims. The advisory forces a SUCCESSFUL verdict, so such a
          // real violation would be silently masked; reject these so a leak /
          // deadlock / race is never hidden behind a dead-code run (issue
          // #4495). Both race spellings must be listed: process_goto_program
          // treats --data-races-check-only as a request to add race assertions
          // too.
          "memory-leak-check",
          "deadlock-check",
          "data-races-check",
          "data-races-check-only"})
      if (cmdline.isset(incompatible))
      {
        log_error(
          "--dead-code-check cannot be combined with --{}", incompatible);
        return true;
      }

  // Combined mode owns the whole pipeline: process_goto_program routes
  // --loop-invariant into goto_loop_invariant_combined, which never reaches
  // the synthesis pass, so the flag would be a silent no-op. Combined mode
  // also ASSUMEs the invariant at the end of the body, and assuming a guess
  // is exactly what synthesis must never do (a wrong candidate has to fail a
  // claim, not license one). Reject rather than wire the two together.
  if (
    cmdline.isset("synthesise-loop-invariants") &&
    cmdline.isset("loop-invariant"))
  {
    log_error(
      "--synthesise-loop-invariants cannot be combined with --loop-invariant; "
      "use --loop-invariant-check, which it implies");
    return true;
  }

  // --termination havocs every loop head k-induction-style (goto_termination),
  // and the standalone schema has already rewritten those heads: goto_loop_
  // invariant uses insert_swap, which leaves the establishment ASSERT in the
  // head's slot. Composing the two is meaningless -- a loop the schema has cut
  // no longer has the iteration behaviour --termination asks about.
  //
  // Combined mode is exempt, and so is --validate-correctness-witness, which
  // routes to it: goto_loop_invariant_combined splices its verification branch
  // *before* the head with destructive_insert, so the head is still the guard
  // GOTO when goto_termination reaches it. Measured on an unbounded loop with
  // a witness-injected loop invariant, where the ranking check does not
  // short-circuit: the run completes.
  for (const char *mode :
       {"synthesise-loop-invariants", "loop-invariant-check"})
    if (cmdline.isset(mode) && cmdline.isset("termination"))
    {
      log_error("--{} cannot be combined with --termination", mode);
      return true;
    }

  if (k_induction_parallel_conflicts(cmdline))
    return true;

  // --incremental-context-bound owns the outer verification loop, re-running
  // do_bmc per context bound; the unwinding strategies each drive an outer
  // loop of their own, so only one driver can own the run (issue #6480).
  if (cmdline.isset("incremental-context-bound"))
    for (const char *incompatible :
         {"termination",
          "incremental-bmc",
          "falsification",
          "k-induction",
          "k-induction-parallel",
          "loop-invariant"})
      if (cmdline.isset(incompatible))
      {
        log_error(
          "--incremental-context-bound cannot be combined with --{}",
          incompatible);
        return true;
      }

  return false;
}

/// The proof cache is wired into the per-claim solve alone, and both its flags
/// are inert without it. Refused rather than accepted, because a run that
/// reuses nothing is indistinguishable from one whose cache is working. The
/// flags are judged on their own terms, so --skip-bmc and the print-and-stop
/// modes are refused too rather than accepting a combination that could never
/// have worked. Called after the GOTO program is built because only by then is
/// the option set final: --multi-property is implied by several flags, the
/// coverage ones not until process_goto_program.
static bool proof_cache_flags_usable(const optionst &options)
{
  const std::string dir = options.get_option("proof-cache");

  if (options.get_bool_option("proof-cache-verify") && dir.empty())
  {
    log_error(
      "--proof-cache-verify has no cache to check without "
      "--proof-cache <dir>");
    return false;
  }

  if (dir.empty())
    return true;

  const std::string why = proof_cache_inactive_reason(options);
  if (!why.empty())
  {
    report_proof_cache_inactive(why);
    return true;
  }

  if (!options.get_bool_option("multi-property"))
  {
    log_error(
      "--proof-cache reuses claims one at a time and needs "
      "--multi-property");
    return false;
  }

  return true;
}

// This is the main entry point of ESBMC. Here ESBMC performs initialisation
// of the algorithms that will be run over the GOTO program at later stages
//
//  1) Parse CMD                            (see "get_command_line_options")
//  2) Create and preprocess a GOTO program (see "get_goto_functions")
//  3) Set user-specified claims            (see "set_claims")
//  4) Perform Bounded Model Checking
//    - Run a particular verification strategy if specified
//      in CMD (see "do_bmc_strategy"), or
//    - Perform a single run of Bounded Model Checking and rely
//      on the simplifier to determine the sufficient verification bound
//      (see "do_bmc")
int esbmc_parseoptionst::run_chosen_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  if (cmdline.isset("incremental-context-bound"))
    return do_context_bound_deepening(options, goto_functions);

  if (cmdline.isset("ts-check"))
    return do_ts_strategy(options, goto_functions);

  if (
    cmdline.isset("termination") || cmdline.isset("incremental-bmc") ||
    cmdline.isset("falsification") || cmdline.isset("k-induction") ||
    cmdline.isset("loop-invariant"))
    return do_bmc_strategy(options, goto_functions);

  // No strategy chosen: rely on the simplifier and the flags set through CMD.
  bmct bmc(goto_functions, options, context);
  return do_bmc(bmc);
}

int esbmc_parseoptionst::doit()
{
  // Configure msg output
  if (cmdline.isset("file-output"))
  {
    FILE *f = fopen(cmdline.getval("file-output"), "w+");
    /* TODO: handle failure */
    out = f;
    messaget::state.out = f;
  }

  // Print a banner with version info to stdout
  {
    FILE *output_stream = messaget::state.out;
    messaget::state.out = stdout;
    log_status(
      "ESBMC version {} {}-bit {} {}",
      ESBMC_VERSION,
      sizeof(void *) * 8,
      config.this_architecture(),
      config.this_operating_system());
    messaget::state.out = output_stream;
  }

  if (cmdline.isset("version"))
    return 0;

  // Unwinding of transition systems
  if (cmdline.isset("module") || cmdline.isset("gen-interface"))
  {
    log_error("This version has no support for hardware modules.");
    return 1;
  }

  if (incompatible_flags(cmdline))
    return 1;

  // Preprocess the input program.
  // (This will not have any effect if OLD_FRONTEND is not enabled.)
  if (cmdline.isset("preprocess"))
  {
    preprocessing();
    return 0;
  }

  // Initialize goto_functions algorithms
  {
    // Loop unrolling
    if (cmdline.isset("goto-unwind") && !cmdline.isset("unwind"))
    {
      size_t unroll_limit = cmdline.isset("unlimited-goto-unwind") ? -1 : 1000;
      goto_preprocess_algorithms.push_back(
        std::make_unique<bounded_loop_unroller>(unroll_limit));
    }

    // Unroll intrinsic support
    goto_preprocess_algorithms.emplace_back(
      std::make_unique<apply_intrinsic_unroller>());

    // Uninitialised-variable check (CWE-457) must run before
    // mark_decl_as_non_det, which would otherwise overwrite every
    // uninitialised DECL with a nondet ASSIGN and erase the property.
    if (cmdline.isset("uninitialised-vars-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_uninit_vars>(context));

    // Unchecked-return-value check (CWE-252). Runs as a preprocessing
    // algorithm so the inserted ASSERTs participate in the same path-
    // condition pruning as the rest of the goto-program.
    if (cmdline.isset("unchecked-return-value-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_unchecked_return>(context));

    // Excessive-allocation-size check (CWE-789). The bound K is the byte
    // limit above which an allocation size is flagged; a bare flag uses the
    // implicit 1 MiB default (see options.cpp).
    if (cmdline.isset("excessive-alloc-check"))
    {
      // boost's value<int> already validated this as an int at parse time (as
      // --unwind / --k-path-coverage do), so atoi only reads it back and never
      // sees the non-numeric input that would make it silently yield 0.
      int k = atoi(cmdline.getval("excessive-alloc-check"));
      if (k <= 0)
      {
        log_error("--excessive-alloc-check=K requires K >= 1 (got {})", k);
        return 1;
      }
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_excessive_alloc>(context, BigInt(k)));
    }

    // Dead-store advisory (CWE-563). Must also run before mark_decl_as_non_det,
    // which rewrites uninitialised DECLs into `DECL; ASSIGN x = nondet` — that
    // synthetic store would otherwise be reported as a spurious dead store.
    if (cmdline.isset("dead-store-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_dead_store>(
          context, dead_store_advisories));

    // Explicitly marking all declared variables as "nondet"
    goto_preprocess_algorithms.emplace_back(
      std::make_unique<mark_decl_as_non_det>(context));

    if (cmdline.isset("function") && cmdline.isset("assign-param-nondet"))
    {
      // assign parameters to "nondet"
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<assign_params_as_non_det>(context));
    }
  }

  // Run this before the main flow. This method performs its own
  // parsing and preprocessing.
  // This is an old implementation of parallel k-induction algorithm.
  // Eventually we will modify it and implement parallel version for all
  // available strategies. Just run it first before everything else
  // for now.
  //
  // The parallel driver has no termination interpretation, so --termination
  // takes priority: fall through to the sequential strategy instead (#6031).
  if (cmdline.isset("k-induction-parallel") && !cmdline.isset("termination"))
    return doit_k_induction_parallel();

  // Parse ESBMC options (CMD + set internal options)
  optionst options;
  get_command_line_options(options);

  // for solidity: detect .sol files in positional args or via --sol
  {
    bool is_solidity = cmdline.isset("sol");
    if (!is_solidity)
    {
      for (const auto &arg : cmdline.args)
      {
        if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".sol")
        {
          is_solidity = true;
          break;
        }
      }
    }
    if (is_solidity)
    {
      // Mark the run as Solidity so downstream analyses (e.g.
      // pointer-analysis VSA) can gate Solidity-specific precision
      // tweaks off this flag without reparsing cmdline.args. Mirrored
      // into the global `config.options` because analyses outside BMC's
      // local `options` scope (static analysers, value_set_domain's
      // transform, etc.) read from the config.
      //
      // Critical: this is a *distinct* option from the user-facing `sol`,
      // which is a string-valued path (--sol <contract.sol>). Reusing the
      // same name would overwrite the user's path with the string "1"
      // (boolean-true serialised into a string option), corrupting
      // contract_path downstream and — among other things — wiping out
      // the contents buffer used for line-number computation, collapsing
      // every Solidity source location to line 1.
      options.set_option("solidity-mode", true);
      config.options.set_option("solidity-mode", true);
      options.set_option(
        "no-align-check", true); // no need to check alignment in solidity
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option(
        "force-malloc-success", true); // for calloc in the 'newexpression'

      // Solidity contracts are wrapped in a `while(nondet) dispatch()`
      // harness whose unwinding assertion can never be proven (the
      // condition is non-deterministic, so the loop has no static
      // bound).  As a result the k-induction forward-condition phase
      // always returns "unable to prove" for every k, burning solver
      // budget across all k iterations without contributing to a
      // proof — only the inductive step can close the verification.
      // Auto-disable forward condition in dispatcher mode (non-
      // `--function`) to recover that budget.
      // `--function` verifies a single function whose internal loops
      // CAN be bounded (`for (i=0; i<N; ++i)`), so leave forward
      // enabled there.
      if (
        !cmdline.isset("function") &&
        !cmdline.isset("enable-forward-condition"))
        options.set_option("disable-forward-condition", true);

      // `--sol <path>` is documented (in options.cpp) as equivalent to
      // a positional argument. boost::program_options parses `<path>`
      // as the VALUE of `--sol`, which does not reach `cmdline.args`;
      // the downstream `create_goto_program` sees an empty positional
      // list and bails with "Please provide a program to verify". Lift
      // the value into `cmdline.args` only when the positional list is
      // empty: the regression suite uses `--sol <display-name>
      // contract.solast` to supply a source-mapping name alongside a
      // pre-compiled AST, so we must not disturb that form (in which
      // case `cmdline.args` already has the .solast and `--sol` is a
      // label, not a source file).
      if (
        cmdline.args.empty() && cmdline.isset("sol") &&
        !std::string(cmdline.getval("sol")).empty())
      {
        cmdline.args.push_back(cmdline.getval("sol"));
      }

      // Auto-select the best SMT backend for Solidity when the user did not
      // explicitly ask for one. Z3 is significantly slower than modern QF_BV
      // engines on the 256-bit bit-vector arithmetic pervasive in Solidity
      // (uint256, mappings, etc.), so prefer Bitwuzla / CVC5 / Boolector.
      // The auto-selection also covers incremental modes (k-induction,
      // incremental-bmc, falsification): Solidity contracts model storage
      // as recursive struct datatypes which Z3 sometimes rejects with
      // "datatype is not well-founded" — Bitwuzla handles them correctly.
      const bool user_picked_solver =
        cmdline.isset("z3") || cmdline.isset("cvc5") ||
        cmdline.isset("bitwuzla") || cmdline.isset("boolector") ||
        cmdline.isset("yices") || cmdline.isset("mathsat") ||
        cmdline.isset("cvc4") || cmdline.isset("smtlib") ||
        cmdline.isset("neurosym") || cmdline.isset("default-solver");

      // Nested-dynamic-array shapes (`T[][]`, `T[N][][M]`, etc.) hit the
      // Phase-0 bare-smt_sort abort under default Bitwuzla because the
      // tuple flattener cannot represent `array<array<tuple, N>, ...>`.
      // CVC5 with `--cvc5-native-tuples` (commit 41878f36cb) handles them
      // via native datatype encoding. Detect the pattern in the source by
      // scanning for `[]` immediately followed by `[` — the marker of a
      // nested-dynamic-array dimension. False positives (memory or local
      // arrays) only result in CVC5 being chosen instead of Bitwuzla,
      // which is sound (just potentially slower for non-nested shapes).
      const std::string padded_solvers =
        std::string(" ") + ESBMC_AVAILABLE_SOLVERS + " ";
      const bool cvc5_available =
        padded_solvers.find(" cvc5 ") != std::string::npos;
      bool nested_dyn_detected = false;
      // A scalar-valued ≥3-level nested mapping
      // (`mapping(K1=>mapping(K2=>mapping(K3=>uint256)))`) lowers to a
      // CONST_ARRAY-initialised infinite mapping array that Bitwuzla
      // cannot equate (asymmetric `(= ca freshsym)` from IS-havoc →
      // "Equality over constant arrays not fully supported yet" abort
      // under assertion BMC; k-induction non-convergence under
      // coverage). CVC5 handles every scalar depth cleanly (regression
      // duals nested_mapping_write_{3,4}lvl_uint256_*). The
      // typeIdentifier marker is `t_mapping$_` (NOT `t_array$_`), so it
      // is detected separately below and routed to plain CVC5 (no
      // native-tuples — that flag is array-tuple-encoding-specific).
      // See notes/Results/branch_cov/STAGE5_RESIDUAL_DIAG.md.
      bool deep_mapping_detected = false;
      // Pattern B: --k-induction with multi-contract dispatch.  The
      // generated $call#0 / $call#1 dispatcher (solidity_convert_call.cpp
      // around line 2777+ / 3135+) emits a sequential O(N) if-else
      // chain on `this->$address`.  Under k-induction iteration this
      // chain is replicated per step, producing a linear chain of
      // 256-bit BV equalities that Bitwuzla's BV-quantifier engine
      // balloons on (see src/solidity-frontend/README.md:564 and memory
      // reference_cvc5_vs_bitwuzla_eoa.md).  CVC5's array+datatype
      // engine handles the chain in seconds.
      //
      // The trigger requires multi-contract AND value-call AND
      // (--bound OR --reentry-check): the dispatcher chain on
      // `this->$address` and the EOA-balance linear scan are only
      // materialized under bounded inter-contract modelling. Without
      // --bound / --reentry-check, ESBMC nondeterministically models
      // external calls and there is no chain to be amplified by
      // k-induction iteration, so CVC5's array+datatype advantage
      // evaporates and Bitwuzla (or the default fallback) handles the
      // residual VC at parity. A multi-contract test without
      // value-call (e.g. inheritance with direct method calls)
      // doesn't materialize the EOA-fallback path, so the dispatcher
      // chain stays small and Bitwuzla is fine. Conversely, a
      // value-call test without --k-induction is BMC-mode and either
      // has too few EOA scans for the chain to matter (single-EOA:
      // Bitwuzla wins) or is best handled with explicit --cvc5
      // (multi-EOA UNSAT: case-by-case). See memory
      // reference_solidity_solver_auto_hint.md "Why NOT auto-route
      // bare value-call".
      int contract_decl_count = 0;
      bool value_call_detected = false;
      if (!user_picked_solver && cvc5_available)
      {
        // Detection strategy:
        //
        //   (1) Prefer a `.solast` file in cmdline.args — it carries
        //       precise `typeIdentifier` strings.  Scan for three
        //       consecutive `t_array$_t_array$_t_array$_` substrings,
        //       the unambiguous marker of ≥3-dimensional array types
        //       (where the flattener produces bare smt_sorts that
        //       trip the Phase-0 abort under default Bitwuzla).
        //
        //   (2) Fall back to `.sol` source scanning only if no
        //       `.solast` is present.  Source-level bracket counting
        //       can't reliably distinguish type declarations from
        //       index expressions, so the fallback is conservative —
        //       it requires three consecutive `[]` (truly empty
        //       brackets, the most reliable type-level marker).
        const std::string ext_solast = ".solast";
        const std::string ext_sol = ".sol";
        std::string solast_path, sol_path;
        for (const auto &arg : cmdline.args)
        {
          if (
            arg.size() >= ext_solast.size() &&
            arg.compare(
              arg.size() - ext_solast.size(), ext_solast.size(), ext_solast) ==
              0)
          {
            solast_path = arg;
          }
          else if (
            arg.size() >= ext_sol.size() &&
            arg.compare(arg.size() - ext_sol.size(), ext_sol.size(), ext_sol) ==
              0)
          {
            sol_path = arg;
          }
        }

        const std::string &scan_path =
          !solast_path.empty() ? solast_path : sol_path;
        const bool scanning_solast = !solast_path.empty();

        std::ifstream f(scan_path);
        if (f.is_open())
        {
          static const std::string ta_marker = "t_array$_";
          int ta_run = 0;
          size_t ta_match_pos = 0;
          static const std::string tm_marker = "t_mapping$_";
          int tm_run = 0;
          size_t tm_match_pos = 0;
          // A struct-VALUED nested mapping (≥2 levels) needs CVC5 too: its
          // K≥2 array-of-struct zero-init lowers to per-field const-arrays
          // Bitwuzla cannot equate.  `t_struct$_` trails the `t_mapping$_`
          // markers within the same typeIdentifier.
          static const std::string ts_marker = "t_struct$_";
          size_t ts_match_pos = 0;
          int empty_bracket_run = 0;
          static const std::string mp_marker = "mapping(";
          size_t mp_match_pos = 0;
          int mapping_chain_run = 0; // .sol: consecutive `mapping(`
          int prev_emit = -1; // last non-whitespace char (.sol scan only)
          bool in_line = false, in_block = false, in_str = false;
          char str_quote = '\0';
          int c;
          while ((c = f.get()) != EOF)
          {
            char ch = static_cast<char>(c);
            // Comment/string stripping only for .sol source — .solast
            // is JSON where typeIdentifier values live INSIDE double
            // quotes; stripping them would skip the markers we need.
            if (!scanning_solast)
            {
              if (in_line)
              {
                if (ch == '\n')
                  in_line = false;
                continue;
              }
              if (in_block)
              {
                if (ch == '*' && f.peek() == '/')
                {
                  f.get();
                  in_block = false;
                }
                continue;
              }
              if (in_str)
              {
                if (ch == '\\')
                {
                  f.get();
                  continue;
                }
                if (ch == str_quote)
                  in_str = false;
                continue;
              }
              if (ch == '/' && f.peek() == '/')
              {
                f.get();
                in_line = true;
                continue;
              }
              if (ch == '/' && f.peek() == '*')
              {
                f.get();
                in_block = true;
                continue;
              }
              if (ch == '"' || ch == '\'')
              {
                in_str = true;
                str_quote = ch;
                continue;
              }
            }

            if (scanning_solast)
            {
              // Sliding match on `t_array$_` markers in
              // typeIdentifier strings.  Three consecutive markers
              // ⇒ array of array of array of ... (≥3 nesting).
              if (ch == ta_marker[ta_match_pos])
              {
                ++ta_match_pos;
                if (ta_match_pos == ta_marker.size())
                {
                  ++ta_run;
                  ta_match_pos = 0;
                  if (ta_run >= 3)
                  {
                    nested_dyn_detected = true;
                    break;
                  }
                }
              }
              else
              {
                ta_match_pos = (ch == ta_marker[0]) ? 1 : 0;
                if (ta_match_pos == 0)
                  ta_run = 0;
              }
              // Count `t_mapping$_` markers WITHIN ONE typeIdentifier
              // JSON string.  A nested mapping
              // `mapping(K1=>mapping(K2=>mapping(K3=>V)))` lowers to a
              // single typeIdentifier `t_mapping$_<K1>_$_t_mapping$_
              // <K2>_$_t_mapping$_<V>...` — the markers are SEPARATED by
              // key/value types (unlike the array case's adjacent
              // `t_array$_t_array$_`), so the run must NOT reset on the
              // intra-string gap; it resets only at the `"` string
              // boundary.  This confines the count to one type, so
              // three *separate* 1-level mappings (three distinct
              // typeIdentifier strings) never reach 3 — only a genuine
              // ≥3-level nested mapping does.
              if (ch == '"')
              {
                tm_run = 0;
                tm_match_pos = 0;
                ts_match_pos = 0;
              }
              else
              {
                // ≥3 nested mapping levels (any leaf): Bitwuzla const-array
                // abort on the scalar-leaf init.
                if (ch == tm_marker[tm_match_pos])
                {
                  ++tm_match_pos;
                  if (tm_match_pos == tm_marker.size())
                  {
                    tm_match_pos = 0;
                    if (++tm_run >= 3)
                    {
                      deep_mapping_detected = true;
                      break;
                    }
                  }
                }
                else
                  tm_match_pos = (ch == tm_marker[0]) ? 1 : 0;

                // ≥2 nested mapping levels with a STRUCT leaf: the K≥2
                // array-of-struct SoA zero-init emits per-field const-arrays
                // Bitwuzla cannot equate — route to CVC5 as well.
                if (ch == ts_marker[ts_match_pos])
                {
                  ++ts_match_pos;
                  if (ts_match_pos == ts_marker.size())
                  {
                    ts_match_pos = 0;
                    if (tm_run >= 2)
                    {
                      deep_mapping_detected = true;
                      break;
                    }
                  }
                }
                else
                  ts_match_pos = (ch == ts_marker[0]) ? 1 : 0;
              }
            }
            else
            {
              // .sol fallback: count CONSECUTIVE empty brackets
              // `[][][]` etc.  Three-or-more in a row marks a real
              // 3+D type declaration; index expressions like
              // `a[i][j]` or `a[0][0][0]` have non-empty inner brackets
              // and don't bump the run.  Whitespace between `]` and
              // `[` is allowed.
              if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
                continue;
              if (prev_emit == '[' && ch == ']')
              {
                ++empty_bracket_run;
                if (empty_bracket_run >= 3)
                {
                  nested_dyn_detected = true;
                  break;
                }
              }
              else if (ch != '[' || prev_emit != ']')
              {
                // Not part of `]\[` or `[]` chain → reset.
                if (!(prev_emit == ']' && ch == '['))
                  empty_bracket_run = 0;
              }
              // .sol fallback for ≥3-level nested mapping: count
              // `mapping(` tokens in a contiguous run.  A real nested
              // mapping `mapping(K1=>mapping(K2=>mapping(K3=>V)))`
              // never contains `;{}` between its `mapping(` tokens, so
              // those reset the run.  Conservative over-approx (perf-
              // only false positives, acceptable per this detector's
              // documented design); the .solast arm is the precise
              // path the regression suite exercises.
              if (ch == ';' || ch == '{' || ch == '}')
              {
                mapping_chain_run = 0;
                mp_match_pos = 0;
              }
              else if (ch == mp_marker[mp_match_pos])
              {
                ++mp_match_pos;
                if (mp_match_pos == mp_marker.size())
                {
                  mp_match_pos = 0;
                  if (++mapping_chain_run >= 3)
                  {
                    deep_mapping_detected = true;
                    break;
                  }
                }
              }
              else
                mp_match_pos = (ch == mp_marker[0]) ? 1 : 0;
              prev_emit = ch;
            }
          }
        }

        // Pattern B contract count: prefer .sol (text-greppable with
        // word-boundary check); fall back to .solast (JSON) when .sol
        // isn't given as a positional argument — which happens when
        // testing_tool.py invokes ESBMC with `--sol contract.sol`
        // (a flag, not positional) plus `contract.solast` (positional).
        if (sol_path.empty() && !solast_path.empty())
        {
          // .solast fallback: each contract definition appears as
          // `"contractKind":"contract"` in the JSON.  No comment/string
          // filtering needed — JSON has neither, and the marker only
          // appears as a key/value pair.  Note: this counts both regular
          // contracts and abstract contracts (both use "contract"), but
          // not interfaces (`"contractKind":"interface"`) or libraries
          // (`"contractKind":"library"`) — matching .sol semantics.
          std::ifstream lf(solast_path);
          if (lf.is_open())
          {
            static const std::string p_kind = "\"contractKind\":\"contract\"";
            // Value-call markers in solidity solast typeIdentifier
            // strings: transfer/send for built-in address calls;
            // barecall_payable for low-level `.call{value:}`.
            static const std::string p_xfer = "t_function_transfer";
            static const std::string p_send_id = "t_function_send";
            static const std::string p_bare = "t_function_barecall_payable";
            size_t mp_kind = 0, mp_xfer = 0, mp_send = 0, mp_bare = 0;
            auto step =
              [](char ch, size_t &pos, const std::string &pat) -> bool {
              if (ch == pat[pos])
                ++pos;
              else
                pos = (ch == pat[0]) ? 1 : 0;
              if (pos == pat.size())
              {
                pos = 0;
                return true;
              }
              return false;
            };
            int c;
            while ((c = lf.get()) != EOF)
            {
              char ch = static_cast<char>(c);
              if (step(ch, mp_kind, p_kind))
                ++contract_decl_count;
              if (step(ch, mp_xfer, p_xfer))
                value_call_detected = true;
              if (step(ch, mp_send, p_send_id))
                value_call_detected = true;
              if (step(ch, mp_bare, p_bare))
                value_call_detected = true;
            }
          }
        }
        else if (!sol_path.empty())
        {
          std::ifstream sf(sol_path);
          if (sf.is_open())
          {
            static const std::string p_contract = "contract";
            static const std::string p_xfer = ".transfer(";
            static const std::string p_send_d = ".send(";
            static const std::string p_call = ".call{";
            size_t mp_contract = 0;
            size_t mp_xfer = 0, mp_send = 0, mp_call = 0;
            char prev_for_contract = '\0';
            bool sin_line = false, sin_block = false, sin_str = false;
            char sstr_quote = '\0';
            int sc;
            while ((sc = sf.get()) != EOF)
            {
              char ch = static_cast<char>(sc);
              if (sin_line)
              {
                if (ch == '\n')
                  sin_line = false;
                continue;
              }
              if (sin_block)
              {
                if (ch == '*' && sf.peek() == '/')
                {
                  sf.get();
                  sin_block = false;
                }
                continue;
              }
              if (sin_str)
              {
                if (ch == '\\')
                {
                  sf.get();
                  continue;
                }
                if (ch == sstr_quote)
                  sin_str = false;
                continue;
              }
              if (ch == '/' && sf.peek() == '/')
              {
                sf.get();
                sin_line = true;
                continue;
              }
              if (ch == '/' && sf.peek() == '*')
              {
                sf.get();
                sin_block = true;
                continue;
              }
              if (ch == '"' || ch == '\'')
              {
                sin_str = true;
                sstr_quote = ch;
                continue;
              }

              // Value-call detection (sliding match, no boundary check
              // needed — the leading `.` is the boundary).
              auto step = [ch](size_t &pos, const std::string &pat) -> bool {
                if (ch == pat[pos])
                  ++pos;
                else
                  pos = (ch == pat[0]) ? 1 : 0;
                if (pos == pat.size())
                {
                  pos = 0;
                  return true;
                }
                return false;
              };
              if (step(mp_xfer, p_xfer))
                value_call_detected = true;
              if (step(mp_send, p_send_d))
                value_call_detected = true;
              if (step(mp_call, p_call))
                value_call_detected = true;

              // `contract` keyword count: must be a whole word —
              // preceded by non-identifier char (or start-of-file) AND
              // followed by non-identifier char.  Filters out usage as
              // an identifier substring (rare but possible in import
              // paths etc., already string-stripped above).
              if (mp_contract == 0)
              {
                bool boundary =
                  (prev_for_contract == '\0' ||
                   !(std::isalnum(
                       static_cast<unsigned char>(prev_for_contract)) ||
                     prev_for_contract == '_'));
                if (boundary && ch == p_contract[0])
                  mp_contract = 1;
              }
              else if (ch == p_contract[mp_contract])
              {
                ++mp_contract;
                if (mp_contract == p_contract.size())
                {
                  int nx = sf.peek();
                  bool word_end =
                    (nx == EOF ||
                     !(std::isalnum(static_cast<unsigned char>(nx)) ||
                       nx == '_'));
                  if (word_end)
                    ++contract_decl_count;
                  mp_contract = 0;
                }
              }
              else
              {
                mp_contract = (ch == p_contract[0]) ? 1 : 0;
              }

              prev_for_contract = ch;
            }
          }
        }
      }

      // Pattern B fires only when ALL four signals align:
      //   (1) k-induction iteration multiplier
      //   (2) ≥2 contracts (dispatcher chain materializes)
      //   (3) value-call (transfer/send/.call{value:}) — the EOA
      //       fallback path that makes the chain hot
      //   (4) --bound or --reentry-check: the `this->$address` chain
      //       and the EOA-balance linear scan are only materialized
      //       under bounded inter-contract modelling. Without either,
      //       external calls are nondet, the chain doesn't exist, and
      //       CVC5's array+datatype advantage evaporates — Bitwuzla
      //       (or the default fallback order) handles the residual
      //       VC at parity. (--reentry-check internally enables
      //       --bound through the reentry harness.)
      // Without any of (1)..(4), Bitwuzla handles the workload fine.
      const bool kind_multi_contract_detected =
        cmdline.isset("k-induction") &&
        (cmdline.isset("bound") || cmdline.isset("reentry-check")) &&
        contract_decl_count >= 2 && value_call_detected;

      if (!user_picked_solver)
      {
        const char *chosen = nullptr;
        if (nested_dyn_detected)
        {
          // Force CVC5 for nested-dyn-array shapes (Bitwuzla aborts).
          chosen = "cvc5";
          if (!cmdline.isset("no-cvc5-native-tuples"))
            options.set_option("cvc5-native-tuples", true);
        }
        else if (deep_mapping_detected)
        {
          // ≥3-level nested mapping with a scalar leaf.  Bitwuzla
          // aborts on the CONST_ARRAY-initialised infinite mapping
          // array; CVC5's array engine handles it.  Plain CVC5 — NO
          // native-tuples (that is array-tuple-encoding-specific;
          // mirrors Pattern B's plain-CVC5 selection below).
          chosen = "cvc5";
        }
        else if (kind_multi_contract_detected)
        {
          // Pattern B: k-induction-amplified linear if-else chain on
          // 256-bit address equality.  Bitwuzla balloons; CVC5 fast.
          // No native-tuples — that's nested-array-specific encoding.
          chosen = "cvc5";
        }
        else
        {
          const char *preferred[] = {"bitwuzla", "cvc5", "boolector", "z3"};
          for (const char *name : preferred)
          {
            if (
              padded_solvers.find(std::string(" ") + name + " ") !=
              std::string::npos)
            {
              chosen = name;
              break;
            }
          }
        }
        if (chosen)
        {
          options.set_option("default-solver", chosen);
          if (nested_dyn_detected)
          {
            const bool native_on = !cmdline.isset("no-cvc5-native-tuples");
            log_status(
              "Solidity: detected nested-dynamic-array shape; auto-selecting "
              "'cvc5'{} (Bitwuzla's tuple flattener cannot encode "
              "`T[][]`-style storage). Override with --bitwuzla / --z3 "
              "or pass --no-cvc5-native-tuples to disable native-tuple "
              "encoding.",
              native_on ? " with --cvc5-native-tuples" : "");
          }
          else if (deep_mapping_detected)
          {
            log_status(
              "Solidity: detected >=3-level nested-mapping shape (or a "
              ">=2-level struct-valued mapping); auto-selecting 'cvc5' "
              "(Bitwuzla aborts on the CONST_ARRAY-initialised infinite "
              "mapping array — \"Equality over constant arrays not fully "
              "supported\" under assertion BMC, k-induction non-convergence "
              "under coverage). Override with --bitwuzla / --z3 / "
              "--boolector.");
          }
          else if (kind_multi_contract_detected)
          {
            const char *bound_reason =
              cmdline.isset("bound") ? "--bound" : "--reentry-check";
            log_status(
              "Solidity: detected --k-induction with multi-contract "
              "dispatch ({} contracts) AND value-routing call "
              "(.transfer/.send/.call{{value:}}) under {}; auto-selecting "
              "'cvc5' (Bitwuzla balloons on the k-induction-amplified "
              "linear if-else chain over 256-bit address equality "
              "materialized in bounded inter-contract mode). Override "
              "with --bitwuzla / --z3 / --boolector.",
              contract_decl_count,
              bound_reason);
          }
          else
          {
            log_status(
              "Solidity: auto-selecting '{}' as SMT backend (Z3 is much "
              "slower on 256-bit bit-vector arithmetic). Override with "
              "--z3 / --cvc5 / --bitwuzla / --boolector or --default-solver.",
              chosen);
          }
        }
      }
    }
  }

#ifdef ENABLE_SOLIDITY_FRONTEND
  // TOD harness generation is driven by one of two flags:
  //   --tod-balance-check  -> balance-TOD: pair shares address(this).balance
  //   --tod-race-check     -> storage-race TOD: pair shares a non-balance var
  // Each flag takes "auto" (discover pairs) or "f1,f2" (specific pair).
  if (cmdline.isset("tod-balance-check") || cmdline.isset("tod-race-check"))
  {
    if (cmdline.isset("tod-balance-check") && cmdline.isset("tod-race-check"))
    {
      log_error(
        "--tod-balance-check and --tod-race-check are mutually exclusive");
      return 1;
    }
    const bool balance_mode = cmdline.isset("tod-balance-check");
    const char *tod_flag =
      balance_mode ? "tod-balance-check" : "tod-race-check";
    const solidity_tod::Mode tod_mode = balance_mode
                                          ? solidity_tod::Mode::BalanceOnly
                                          : solidity_tod::Mode::RaceOnly;
    const char *mode_tag = balance_mode ? "balance" : "race";
    // ---- shared: locate .sol / .solast paths ----
    std::string sol_path, solast_path;
    if (cmdline.isset("sol"))
      sol_path = cmdline.getval("sol");
    for (const auto &arg : cmdline.args)
    {
      if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".sol")
      {
        if (sol_path.empty())
          sol_path = arg;
      }
      else if (arg.size() >= 7 && arg.substr(arg.size() - 7) == ".solast")
        solast_path = arg;
    }
    if (sol_path.empty())
    {
      log_error("--{} requires a .sol source file", tod_flag);
      return 1;
    }

    std::ifstream sol_file(sol_path);
    if (!sol_file.is_open())
    {
      log_error("Cannot open source file: {}", sol_path);
      return 1;
    }
    std::string sol_source(
      (std::istreambuf_iterator<char>(sol_file)),
      std::istreambuf_iterator<char>());

    if (solast_path.empty())
    {
      solast_path = sol_path + "ast";
      std::string cmd =
        "solc --ast-compact-json " + sol_path + " > " + solast_path;
      if (cmdline.isset("solc-bin"))
        cmd = std::string(cmdline.getval("solc-bin")) + " --ast-compact-json " +
              sol_path + " > " + solast_path;
      if (system(cmd.c_str()) != 0)
      {
        log_error("solc failed: {}", cmd);
        return 1;
      }
    }

    std::ifstream ast_file(solast_path);
    if (!ast_file.is_open())
    {
      log_error("Cannot open AST file: {}", solast_path);
      return 1;
    }
    nlohmann::json ast;
    {
      std::string line, json_block;
      while (getline(ast_file, line))
        if (line.find(".sol =======") != std::string::npos)
          break;
      while (getline(ast_file, line))
      {
        if (line.find(".sol =======") != std::string::npos)
          break;
        json_block += line + "\n";
      }
      if (json_block.empty())
      {
        log_error("No JSON block found in {}", solast_path);
        return 1;
      }
      ast = nlohmann::json::parse(json_block);
    }

    std::string contract_name;
    if (cmdline.isset("contract"))
      contract_name = cmdline.getval("contract");
    if (contract_name.empty())
    {
      log_error("TOD: --contract is required to identify the target contract");
      return 1;
    }

    // ---- decide pair list ----
    std::vector<std::pair<std::string, std::string>> pairs;
    std::string tod_val = cmdline.getval(tod_flag);
    bool auto_mode = (tod_val == "auto");

    if (auto_mode)
    {
      const nlohmann::json *cdef = nullptr;
      if (ast.contains("nodes"))
        for (const auto &n : ast["nodes"])
          if (
            n.value("nodeType", "") == "ContractDefinition" &&
            n.value("name", "") == contract_name)
          {
            cdef = &n;
            break;
          }
      if (!cdef)
      {
        log_error(
          "--{}: contract '{}' not found in AST", tod_flag, contract_name);
        return 1;
      }
      auto candidates =
        solidity_tod::find_tod_candidates(*cdef, tod_mode, &ast);
      log_status(
        "--{}: discovered {} candidate pair(s) in '{}'",
        tod_flag,
        candidates.size(),
        contract_name);
      for (const auto &p : candidates)
      {
        log_status("  - {} vs {}", p.func_a, p.func_b);
        pairs.emplace_back(p.func_a, p.func_b);
      }
      if (pairs.empty())
      {
        log_status(
          "--{}: no {} TOD pairs detected — done.", tod_flag, mode_tag);
        return 0;
      }
    }
    else
    {
      auto comma = tod_val.find(',');
      if (comma == std::string::npos)
      {
        log_error(
          "--{} expects 'auto' or two comma-separated function names",
          tod_flag);
        return 1;
      }
      pairs.emplace_back(tod_val.substr(0, comma), tod_val.substr(comma + 1));
    }

    // Each pair -> its own .sol file (one TOD_<a>_<b> contract per file).
    TodHarnessMode harness_mode =
      balance_mode ? TodHarnessMode::Balance : TodHarnessMode::Race;
    std::string sol_dir;
    {
      auto slash = sol_path.find_last_of("/\\");
      sol_dir = (slash == std::string::npos) ? std::string(".")
                                             : sol_path.substr(0, slash);
    }
    auto harness_path_for = [&](const std::string &fa, const std::string &fb) {
      return sol_dir + "/tod_" + mode_tag + "_" + fa + "_" + fb +
             "_harness.sol";
    };

    // --dump-harness: print the first pair and exit.
    if (cmdline.isset("dump-harness"))
    {
      std::string harness = generate_tod_harness(
        sol_source,
        ast,
        contract_name,
        pairs[0].first,
        pairs[0].second,
        harness_mode);
      if (harness.empty())
        return 1;
      std::cout << harness;
      return 0;
    }

    // ---- single pair: generate file, verify in-process ----
    if (!auto_mode)
    {
      std::string harness = generate_tod_harness(
        sol_source,
        ast,
        contract_name,
        pairs[0].first,
        pairs[0].second,
        harness_mode);
      if (harness.empty())
        return 1;
      std::string harness_path =
        harness_path_for(pairs[0].first, pairs[0].second);
      std::ofstream out(harness_path);
      if (!out.is_open())
      {
        log_error("Cannot write harness to {}", harness_path);
        return 1;
      }
      out << harness;
      out.close();
      log_status("TOD harness written to {}", harness_path);

      std::string harness_contract =
        "TOD_" + pairs[0].first + "_" + pairs[0].second;
      cmdline.args.clear();
      cmdline.args.push_back(harness_path);
      config.cname = harness_contract;
      options.set_option("contract", harness_contract);
      options.set_option("bound", true);
      options.set_option("no-standard-checks", true);
      // The generated TOD harness drives two transactions in two orders via
      // the dispatcher loop; it must use the unbounded harness, not the
      // bounded-by-default unroll (which would hide the ordering race).
      options.set_option("solidity-max-tx", "0");
      config.options = options;
    }
    // ---- auto: one .sol per pair, one subprocess per .sol ----
    else
    {
      std::string esbmc = executable_path.string();
      // --solidity-max-tx 0: the TOD harness needs the unbounded dispatcher
      // loop to explore both transaction orderings (bounded-by-default would
      // hide the race).
      std::string forwarded =
        " --bound --no-standard-checks --solidity-max-tx 0";
      if (cmdline.isset("unwind"))
        forwarded += std::string(" --unwind ") + cmdline.getval("unwind");
      if (cmdline.isset("no-unwinding-assertions"))
        forwarded += " --no-unwinding-assertions";
      if (cmdline.isset("incremental-bmc"))
        forwarded += " --incremental-bmc";
      if (cmdline.isset("max-k-step"))
        forwarded +=
          std::string(" --max-k-step ") + cmdline.getval("max-k-step");
      for (const char *flag :
           {"cvc5", "bitwuzla", "boolector", "z3", "yices", "mathsat"})
        if (cmdline.isset(flag))
          forwarded += std::string(" --") + flag;

      // Materialise every harness file first so workers only spawn ESBMC.
      std::vector<std::string> harness_paths(pairs.size());
      std::vector<bool> harness_ok(pairs.size(), false);
      for (size_t i = 0; i < pairs.size(); ++i)
      {
        const auto &p = pairs[i];
        std::string harness = generate_tod_harness(
          sol_source, ast, contract_name, p.first, p.second, harness_mode);
        if (harness.empty())
        {
          log_error(
            "--{}: harness generation failed for {}/{}",
            tod_flag,
            p.first,
            p.second);
          continue;
        }
        std::string hp = harness_path_for(p.first, p.second);
        std::ofstream out(hp);
        if (!out.is_open())
        {
          log_error("Cannot write harness to {}", hp);
          continue;
        }
        out << harness;
        harness_paths[i] = hp;
        harness_ok[i] = true;
      }

      unsigned int jobs = 0;
      if (cmdline.isset("tod-jobs"))
        jobs = std::stoul(cmdline.getval("tod-jobs"));
      if (jobs == 0)
      {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0)
          hw = 1;
        jobs = std::min<unsigned>(hw, pairs.size());
      }
      if (jobs == 0)
        jobs = 1;
      log_status(
        "--{}: launching {} parallel ESBMC job(s) over {} pair(s)",
        tod_flag,
        jobs,
        pairs.size());

      std::atomic<size_t> next_idx{0};
      std::atomic<size_t> fail_count{0}, succ_count{0}, error_count{0};
      std::mutex failed_mutex;
      std::vector<std::string> failed_pairs;

      auto worker = [&]() {
        for (;;)
        {
          size_t i = next_idx.fetch_add(1);
          if (i >= pairs.size())
            return;
          if (!harness_ok[i])
          {
            error_count.fetch_add(1);
            continue;
          }
          const auto &p = pairs[i];
          std::string c = "TOD_" + p.first + "_" + p.second;
          std::string cmd = esbmc + " " + harness_paths[i] + forwarded +
                            " --contract " + c + " 2>&1";
          log_status("--{}: verifying {} ({})", tod_flag, c, harness_paths[i]);
          FILE *fp = popen(cmd.c_str(), "r");
          if (!fp)
          {
            log_error("popen failed for: {}", cmd);
            error_count.fetch_add(1);
            continue;
          }
          std::string output;
          char buf[4096];
          while (fgets(buf, sizeof(buf), fp))
            output += buf;
          pclose(fp);

          if (output.find("VERIFICATION SUCCESSFUL") != std::string::npos)
          {
            log_status("  -> {} SUCCESSFUL", c);
            succ_count.fetch_add(1);
          }
          else if (output.find("VERIFICATION FAILED") != std::string::npos)
          {
            log_fail(
              "  -> {} FAILED  (TOD vulnerability between {} and {})",
              c,
              p.first,
              p.second);
            fail_count.fetch_add(1);
            std::lock_guard<std::mutex> lk(failed_mutex);
            failed_pairs.push_back(c);
          }
          else
          {
            log_error("  -> {} ERROR (no verdict)", c);
            error_count.fetch_add(1);
          }
        }
      };

      std::vector<std::thread> threads;
      threads.reserve(jobs);
      for (unsigned t = 0; t < jobs; ++t)
        threads.emplace_back(worker);
      for (auto &th : threads)
        th.join();

      log_status(
        "--{} summary: {} pair(s) — {} clean, {} TOD found, {} error",
        tod_flag,
        pairs.size(),
        succ_count.load(),
        fail_count.load(),
        error_count.load());
      if (!failed_pairs.empty())
      {
        log_fail("TOD-vulnerable pairs:");
        for (const auto &c : failed_pairs)
          log_fail("  - {}", c);
      }
      return (fail_count.load() > 0 || error_count.load() > 0) ? 1 : 0;
    }
  }
#endif

  // Installed before the GOTO program is built so the check also covers the
  // simplification the frontend and the GOTO passes perform. Inert unless
  // the build enabled ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK.
  install_simplification_equivalence_check(namespacet(context), options);
  // Reports and uninstalls on the way out, whichever of doit()'s exits is
  // taken. Uninstalling here rather than at the end of GOTO construction left
  // symex -- where the rewrites that decide a verdict actually happen --
  // unchecked, and the count describing only the frontend (esbmc/esbmc#7260).
  const simplification_check_scopet simplification_check_scope;

  // Create and preprocess a GOTO program
  if (get_goto_program(options, goto_functions))
    return 6;

  // Output claims about this program
  // (Fedor: should be moved to the output method perhaps)
  if (cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  // Set user-specified claims
  // (Fedor: should be moved to the preprocessing method perhaps)
  if (set_claims(goto_functions))
    return 7;

  if (!proof_cache_flags_usable(options))
    return 1;

  // Leave without doing any Bounded Model Checking
  if (options.get_bool_option("skip-bmc"))
    return 0;

  // A violation found under a bounded schedule space is genuine whichever
  // strategy owns the run, so this precedes the dispatch rather than joins it.
  const int prepass = falsify_with_bounded_schedules(options, goto_functions);
  if (prepass >= 0)
    return prepass;

  const int res = run_chosen_strategy(options, goto_functions);

  // Dead-code analysis is advisory: its probes are SAT for every live branch,
  // which do_bmc maps to a non-zero (FAILED) exit code. The findings are
  // reported separately, so remap that to 0 — but only for a completed
  // analysis. A solver error (P_ERROR) or an SMTLIB-only emission (P_SMTLIB)
  // is not a finished advisory run, so propagate it rather than masking a
  // crashed/incomplete analysis as success (issue #4495).
  //
  // Checked before the coverage branch below: `is_coverage` also covers
  // --dead-code-check, which borrows the instrumentation but reports
  // advisories and a verdict rather than a [Coverage] block.
  if (options.get_bool_option("dead-code-check"))
  {
    if (res == P_ERROR || res == P_SMTLIB)
      return res;
    return 0;
  }

  // A coverage run has no verdict, so it has no failure exit code either: its
  // result is the [Coverage] block closed by the completeness line. Printed
  // here, once, so it lands after the last block (k-induction prints one per
  // phase).
  // A path-coverage query keeps the verdict it printed, and its exit code.
  if (is_coverage && !goto_coveraget::path_cov_query_mode())
  {
    report_coverage_completeness();
    return 0;
  }
  return res;
}

bool esbmc_parseoptionst::resolve_color_option() const
{
  const char *raw = cmdline.getval("color");
  std::string val = (raw && *raw) ? raw : "auto";
  if (val != "auto" && val != "always" && val != "never")
  {
    log_error(
      "Invalid value for --color: '{}'. Must be auto, always, or never.", val);
    exit(1);
  }
  return ENABLE_COLOR(val);
}

// Colorize --flag references found in description text with bold formatting.
// Matches "--" followed by one or more alphanumeric/hyphen characters,
// stopping at delimiters like '.', ',', ' ', ')', '\'', '"', or end of string.
static std::string colorize_flag_refs(const std::string &text)
{
  std::string result;
  size_t i = 0;
  while (i < text.size())
  {
    if (
      i + 2 < text.size() && text[i] == '-' && text[i + 1] == '-' &&
      (std::isalnum((unsigned char)text[i + 2]) || text[i + 2] == '-'))
    {
      size_t start = i;
      i += 2;
      while (i < text.size() &&
             (std::isalnum((unsigned char)text[i]) || text[i] == '-'))
        i++;
      result += CLR_BOLD;
      result += text.substr(start, i - start);
      result += CLR_RESET;
    }
    else
    {
      result += text[i];
      i++;
    }
  }
  return result;
}

// This prints the ESBMC version and a list of CMD options
// available in ESBMC.
void esbmc_parseoptionst::help()
{
  // Redirect everything here to stdout
  FILE *outstream = messaget::state.out;
  messaget::state.out = stdout;

  bool use_color = resolve_color_option();

  // Print the "* * *     ESBMC x.y.z     * * *"
  auto const esbmc_string = fmt::format(" ESBMC {} ", ESBMC_VERSION);
  auto const title_start = std::string("* * * ");
  auto const title_end = std::string(" * * *");
  // Clamp at 0 so an over-long version string can't underflow the
  // unsigned padding width.
  auto const inner = std::max<ptrdiff_t>(
    0,
    80 - static_cast<ptrdiff_t>(
           title_start.length() + title_end.length() + esbmc_string.length()));
  auto const left_pad = std::string(inner / 2, '=');
  auto const right_pad = std::string(inner - inner / 2, '=');
  log_status(
    "\n{}{}{}{}{}", title_start, left_pad, esbmc_string, right_pad, title_end);

  std::ostringstream oss;
  oss << cmdline.cmdline_options;

  if (!use_color)
  {
    log_status("{}", oss.str());
    return;
  }

  // Colorize: group headers in bold cyan, option names in bold,
  // and --flag references in descriptions in bold
  std::istringstream iss(oss.str());
  std::string line;
  while (std::getline(iss, line))
  {
    if (!line.empty() && line[0] != ' ' && line.back() == ':')
      // Group header (e.g. "Printing options:")
      fmt::print(messaget::state.out, CLR_BOLD_CYAN "{}" CLR_RESET "\n", line);
    else if (
      line.size() >= 3 && line[0] == ' ' && line[1] == ' ' && line[2] == '-')
    {
      // Option line: colorize the flag portion (up to the description)
      auto desc_pos = line.find("  ", 4);
      if (desc_pos != std::string::npos)
        fmt::print(
          messaget::state.out,
          CLR_BOLD "{}" CLR_RESET "{}\n",
          line.substr(0, desc_pos),
          colorize_flag_refs(line.substr(desc_pos)));
      else
        fmt::print(messaget::state.out, CLR_BOLD "{}" CLR_RESET "\n", line);
    }
    else
      fmt::print(messaget::state.out, "{}\n", colorize_flag_refs(line));
  }

  // Restore everything back to original output stream.
  messaget::state.out = outstream;
}
