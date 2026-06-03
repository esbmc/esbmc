#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <goto-symex/symex_target_equation.h>
#include <util/time_stopping.h>
#include <util/algorithms.h>
#include <util/options.h>
#include <boost/range/adaptor/reversed.hpp>
#include <langapi/language_util.h>

/* Base interface */
class slicer : public ssa_step_algorithm
{
public:
  slicer() : ssa_step_algorithm(true)
  {
  }
  BigInt ignored() const override
  {
    return sliced;
  }

protected:
  /// tracks how many steps were sliced
  BigInt sliced = 0;
};

/**
 * Naive slicer: slice every step after the last assertion
 * @param eq symex formula to be sliced
 * @return number of steps that were ignored
 */
class simple_slice : public slicer
{
public:
  simple_slice() = default;
  bool run(symex_target_equationt::SSA_stepst &) override;
};

/**
 * Claim slicer: remove every other claim of the formula
 * with the exception of the one to keep.
 */
class claim_slicer : public slicer
{
public:
  explicit claim_slicer(
    const size_t claim_to_keep,
    bool show_slice_info,
    bool is_goto_cov,
    namespacet &ns)
    : claim_to_keep(claim_to_keep),
      show_slice_info(show_slice_info),
      is_goto_cov(is_goto_cov),
      ns(ns)
  {
    if (!claim_to_keep)
    {
      log_error("All the claims start from 1 (use --show-claims)");
      abort();
    }
  };
  bool run(symex_target_equationt::SSA_stepst &) override;
  size_t claim_to_keep;
  std::string claim_msg;
  std::string claim_loc;
  std::string claim_cstr;
  std::string claim_property;
  bool show_slice_info;
  bool is_goto_cov;
  namespacet ns;
};

/**
 * @brief Class for the symex-slicer, this slicer is to be executed
 * on SSA formula in order to remove every symbol that does not depends
 * on it
 *
 * It works by constructing a symbol dependency list by transversing
 * the SSA formula in reverse order. If any assume, assignment, or renumber
 * step does not belong into this dependency, then it will be ignored.
 */
class symex_slicet : public slicer
{
public:
  explicit symex_slicet(const optionst &options)
    : slice_assumes(options.get_bool_option("slice-assumes")),
      slice_nondet(
        !options.get_bool_option("generate-testcase") &&
        !options.get_bool_option("generate-ctest-testcase"))
  {
  }

  /**
   * Iterate over all steps of the \eq in REVERSE order,
   * getting symbol dependencies. If an
   * assignment, renumber or assume does not contain one
   * of the dependency symbols, then it will be ignored.
   *
   * @param eq symex formula to be sliced
   */
  bool run(symex_target_equationt::SSA_stepst &eq) override
  {
    // The slicer object is reused across equations (e.g. incremental-bmc /
    // k-induction run it once per produced formula). Each formula must be
    // sliced independently, so clear all per-equation state up front.
    // (Previously only `sliced` was reset, leaking `depends` across equations.)
    sliced = 0;
    depends.clear();
    collected_cache.clear();
    index_reads.clear();
    array_disqualified.clear();
    scanned_cache.clear();

    fine_timet algorithm_start = current_time();
    for (auto &step : boost::adaptors::reverse(eq))
    {
      if (step.ignore)
        continue;
      run_on_step(step);
    }
    fine_timet algorithm_stop = current_time();
    log_status(
      "Slicing time: {}s (removed {} assignments)",
      time2string(algorithm_stop - algorithm_start),
      sliced);
    return true;
  }

  /**
   * Holds the symbols the current equation depends on.
   */
  std::unordered_set<std::string> depends;

  /// BLACK set (two-color DFS in collect_dependencies): nodes already fully
  /// collected into #depends. Persists across steps so the shared guard
  /// and-chain prefix is walked once overall (Θ(N²) -> O(N)). Collection is
  /// purely monotone (depends.insert only), so the memo is unconditional.
  std::unordered_set<const expr2t *> collected_cache;

  /// Per-array-version read map: versioned array-symbol name -> set of constant
  /// indices that are actually read. Propagation through a kept store inserts
  /// the live-out set minus the overwritten index, preserving any direct reads
  /// already recorded for the source version. A store to an index NOT present
  /// here is provably dead and may be elided. Populated by #scan_array_uses,
  /// consulted by
  /// #run_on_assignment to drop dead array stores.
  std::unordered_map<std::string, std::unordered_set<size_t>> index_reads;

  /// Array versions whose element reads cannot be reasoned about per-index
  /// (whole-array use, symbolic index, array-as-value, …). INSERT-ONLY. A
  /// disqualified array never has any of its stores dropped.
  std::unordered_set<std::string> array_disqualified;

  /// BLACK set for the #scan_array_uses two-color DFS — independent of
  /// #collected_cache. Sound for the same reason: index_reads/array_disqualified
  /// are insert-only, so re-skipping an already-scanned subtree adds nothing.
  std::unordered_set<const expr2t *> scanned_cache;

  static expr2tc get_nondet_symbol(const expr2tc &expr);

  /**
   * Marks SSA_steps to be ignored which have no effects on the target equation,
   * according to the options set in the `config`.
   *
   * Notably, this function depends on the global `config`:
   *  - "no-slice" in `options` -> perform only simple slicing: ignore everything
   *    after the final assertion
   *  - "slice-assumes" in `options` -> also perform slicing of assumption steps
   *  - `config.no_slice_names` and `config.no_slice_ids` -> suppress slicing of
   *    particular symbols in non-simple slicing mode.
   *
   *    * Note 1: ASSERTS are not sliced, only their symbols are added
   * into the #depends
   *
   *    * Note 2: Similar to ASSERTS, if 'slice-assumes' option is
   * is not enabled. Then only its symbols are added into the
   * #depends

   *
   * @param eq The target equation containing the SSA steps to perform program
   *           slicing on.
   * @return Whether any step was sliced away
   */

protected:
  /// whether assumes should be sliced
  const bool slice_assumes;
  /// Whether we should slice nondet symbols
  const bool slice_nondet;

  /**
   * Collect every symbol of \expr into #depends (mutating "reverse-taint"
   * accumulation). Walked with an explicit, memoised worklist — see the
   * implementation for the two-color DFS that keeps this O(N) and stack-safe
   * over deep guard and-chains.
   *
   * @param expr expression whose symbols are added to #depends
   */
  void collect_dependencies(const expr2tc &expr);

  /**
   * Record the array-element reads in \expr into #index_reads, and disqualify
   * (#array_disqualified) any array version used in a way that defeats
   * per-index reasoning. Run on the guard/cond/rhs of every RETAINED step so
   * that, by the time a store is examined in reverse order, every downstream
   * read of that array version has already been recorded.
   *
   * Walked with the same explicit two-color worklist as #collect_dependencies
   * but with its own memo (#scanned_cache); both outputs are insert-only, so
   * skipping an already-scanned subtree is sound and keeps the scan O(N).
   *
   * @param expr expression whose array reads are recorded
   */
  void scan_array_uses(const expr2tc &expr);

  /**
   * Read-only: does \expr reference a tracked dependency (a symbol in #depends
   * or a no_slice symbol)? Used only on shallow exprs (an SSA lhs symbol, or an
   * assume cond), never the deep guard, so it stays recursive and is NOT
   * memoised — #depends mutates across steps, so a node-memo would be unsound.
   *
   * @param expr expression to test
   * @return true if \expr touches the tracked dependency set
   */
  bool depends_on_tracked(const expr2tc &expr);

  /**
   * Remove unneeded assumes from the formula
   *
   * Check if the Assume cond symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * Note 1: All the conditions operands are going to be added
   * into the #depends. This makes that the condition itself as
   * a "reverse taint"
   *
   * TODO: What happens if the ASSUME would result in false?
   *
   * @param SSA_step an assume step
   */
  void run_on_assert(symex_target_equationt::SSA_stept &SSA_step) override;

  /**
   * Remove unneeded assumes from the formula
   *
   * Check if the Assume cond symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * Note 1: All the conditions operands are going to be added
   * into the #depends. This makes that the condition itself as
   * a "reverse taint"
   *
   * TODO: What happens if the ASSUME would result in false?
   *
   * @param SSA_step an assume step
   */
  void run_on_assume(symex_target_equationt::SSA_stept &SSA_step) override;

  /**
   * Remove unneeded assignments from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an assignment step
   */
  void run_on_assignment(symex_target_equationt::SSA_stept &SSA_step) override;

  /**
   * Remove unneeded renumbers from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an renumber step
   */
  void run_on_renumber(symex_target_equationt::SSA_stept &SSA_step) override;
};

#endif
