#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

#include <goto-programs/dead_store_advisory.h>
#include <goto-programs/goto_coverage.h>
#include <goto-programs/property_verdict.h>
#include <goto-symex/slice.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <goto-symex/witnesses.h>
#include <goto-symex/pytest.h>
#include <goto-symex/ctest.h>
#include <langapi/language_ui.h>
#include <list>
#include <map>
#include <set>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/config/options.h>
#include <util/ssa/algorithms.h>
#include <util/config/cmdline.h>
#include <atomic>

class bmct
{
public:
  bmct(goto_functionst &funcs, optionst &opts, contextt &_context);

  optionst &options;

  // Dead-store advisories (CWE-563) computed during goto preprocessing. Set by
  // the driver before verification; surfaced in SARIF as note-level results
  // on both the success and failure paths.
  std::vector<dead_store_advisoryt> dead_store_advisories;
  // True once a SARIF document carrying the advisories has been written — from a
  // trace path, or from the --dead-code-check advisory that folds them into its
  // own document — so start_bmc() does not write a second one. There is a single
  // SARIF output path, so a second write would truncate the first.
  bool dead_store_sarif_written = false;
  enum ltl_res
  {
    ltl_res_good,
    ltl_res_succeeding,
    ltl_res_failing,
    ltl_res_bad
  };
  size_t ltl_results_seen[4];

  /* ltl_run_thread returns an ltl_res, or one of these sentinels. A formula
   * with no prefix assertion, or whose monitor preconditions are violated,
   * carries no usable prefix verdict: neither ⊤ nor any lower outcome may be
   * claimed from it (#6547). */
  static constexpr int ltl_res_uninstrumented = -3;

  // Set when an LTL run produced no four-valued verdict; consulted by
  // report_result to report UNKNOWN rather than the ⊤ that an absent or
  // truncated monitor used to produce.
  std::atomic<bool> ltl_uninstrumented{false};

  BigInt interleaving_number;
  BigInt interleaving_failed;

  /** Mirrors reachability_treet::cs_bound_pruned; valid after start_bmc(). */
  bool cs_bound_pruned = false;

  virtual smt_resultt start_bmc();
  virtual smt_resultt run(std::shared_ptr<symex_target_equationt> &eq);
  virtual ~bmct();

protected:
  const contextt &context;
  namespacet ns;

  std::unique_ptr<smt_convt> runtime_solver;
  std::unique_ptr<reachability_treet> symex;
  pytest_generator pytest_gen; // For Python pytest test case generation
  ctest_generator ctest_gen;   // For C/C++ CTest test case generation
  mutable std::atomic<bool> keep_alive_running;
  mutable std::atomic<int> keep_alive_interval;

  virtual smt_resultt
  run_decision_procedure(smt_convt &smt_conv, symex_target_equationt &eq) const;

  // Re-encode `local_eq` in vacuity mode against a fresh solver and check
  // whether the path to each kept claim is reachable. UNSAT means the
  // discharge was vacuous: the path assumptions alone are unsatisfiable.
  smt_resultt check_vacuity(symex_target_equationt &local_eq) const;

  // Set by the vacuity probe when at least one kept claim discharged
  // vacuously; consulted by report_result to map the final verdict from
  // SUCCESSFUL to UNKNOWN. Atomic because multi_property_check writes from
  // parallel job threads.
  std::atomic<bool> vacuity_detected{false};

  virtual void show_program(const symex_target_equationt &eq);
  virtual void report_success();
  virtual void report_failure();
  virtual void report_unknown();
  virtual void keep_alive_function() const;

  virtual void
  error_trace(smt_convt &smt_conv, const symex_target_equationt &eq);

  virtual void successful_trace(const symex_target_equationt &eq);

  virtual void show_vcc(const symex_target_equationt &eq);

  virtual void show_vcc(std::ostream &out, const symex_target_equationt &eq);

  virtual void report_trace(smt_resultt &res, const symex_target_equationt &eq);

  virtual void report_result(smt_resultt &res);

  virtual void
  bidirectional_search(smt_convt &smt_conv, const symex_target_equationt &eq);

  smt_resultt run_thread(std::shared_ptr<symex_target_equationt> &eq);

  /* Not const: the solver that satisfied a prefix assertion is kept in
   * runtime_solver so the verdict can be backed by a counterexample. */
  int ltl_run_thread(symex_target_equationt &equation);

  /// \param truncated_loops how many loops exploration cut off at the
  ///   unwinding bound, carried in from the symex result rather than read back
  ///   from live state, which --schedule has already invalidated (#6423).
  smt_resultt multi_property_check(
    const symex_target_equationt &eq,
    size_t remaining_claims,
    smt_convt &runtime_solver,
    unsigned int truncated_loops);

  std::vector<std::unique_ptr<ssa_step_algorithm>> algorithms;

  void generate_smt_from_equation(
    smt_convt &smt_conv,
    symex_target_equationt &eq) const;

  // for multi-property
  void clear_verified_claims_in_ssa(
    symex_target_equationt &local_eq,
    const claim_slicer &claim,
    const bool &is_goto_cov);
  void clear_verified_claims_in_goto(
    const claim_slicer &claim,
    const bool &is_goto_cov);

  /// Reason why witness enumeration stopped, used in the report footer.
  enum class enumeration_stop_reasont
  {
    Unsat,    // re-solve returned UNSAT — witness set is exhaustive
    CapHit,   // --max-witnesses limit reached
    NoInputs, // first witness had no nondet inputs to block
    Error,    // re-solve returned an error / unknown
    Disabled, // --all-witnesses not set; single witness reported
  };

  /// One enumerated witness for a failing property.
  struct witness_recordt
  {
    size_t ce_index;                                   // global ce_counter slot
    std::vector<collected_nondet_value> nondet_inputs; // input tuple
    goto_tracet trace;                                 // goto-level trace
  };

  /// Multi-witness textual reporter. Machine-readable artifacts (cex,
  /// testcase, html, json, graphml, yaml) are emitted by the caller while
  /// each witness's solver model is still live; this function only handles
  /// the human-readable counterexample output.
  /// \param reachability_trace render as evidence that the location is
  ///   reachable, not as a property violation (coverage runs; issue #6387).
  virtual void report_multi_property_trace(
    const smt_resultt &res,
    const std::vector<witness_recordt> &witnesses,
    enumeration_stop_reasont stop_reason,
    const std::string &msg,
    bool reachability_trace = false);

  void report_coverage_verbose(
    const claim_slicer &claim,
    const std::string &claim_sig,
    const bool &is_assert_cov,
    const bool &is_cond_cov,
    const bool &is_branch_cov,
    const bool &is_branch_func_cov,
    const bool &is_k_path_cov,
    const std::unordered_set<std::string> &reached_claims,
    const std::unordered_multiset<std::string> &reached_mul_claims);

private:
  /// Report each of the program's properties once, with the verdict that
  /// dominates across every thread interleaving explored, followed by a
  /// summary. A phase that decided nothing and does not conclude the run stays
  /// silent, leaving the report to the phase that does.
  void report_property_verdicts(smt_resultt res) const;

  /// Print the property table, grouped by file and function.
  void print_property_rows(
    const std::vector<struct property_rowt> &rows,
    const struct property_countst &counts) const;

  /// Print the "** N of M properties failed, ..." line.
  void
  print_property_summary(size_t total, const struct property_countst &) const;

  /// Render the verdict table as coverage goals rather than properties.
  void report_coverage_goal_verdicts(
    const std::map<std::string, property_resultt> &verdicts) const;

  /// Enter every assertion in \p eq into the verdict table as NotChecked, so
  /// the report covers the whole program rather than only those properties
  /// some phase happened to reach a verdict on (discussion #7023).
  void seed_property_verdicts(const symex_target_equationt &eq) const;

  /// Record Failed for every assertion in \p eq that \p smt_conv's model
  /// falsifies, so the report names them even when the counterexample itself
  /// is suppressed. Call only where a SAT result witnesses a real violation:
  /// an inductive-step or forward-condition model does not.
  void record_violated_properties(
    smt_convt &smt_conv,
    const symex_target_equationt &eq) const;

  /// Source files whose assertions come from ESBMC's own operational models,
  /// so the report can sort them after the user's code. Empty for Python,
  /// where a hidden body does not imply a model (remove_library_assertions).
  std::set<std::string> library_files;

  /// Whether this phase concludes the run, i.e. report_result() will print a
  /// VERIFICATION verdict for \p res rather than deepen the search. The
  /// property report describes the whole run, so an iterative strategy must
  /// not emit one table per k.
  bool reports_final_verdict(smt_resultt res) const;

  /// Whether \p res establishes that *every* property holds, as opposed to a
  /// merely bounded round such as a k-induction base case. Must agree with the
  /// path through report_result() that reaches report_success().
  bool all_properties_proved(smt_resultt res) const;

  static constexpr size_t default_barren_interleaving_budget = 100;

  /// How many consecutive thread interleavings may reach a verdict on nothing
  /// new before exploration stops after a violation
  /// (--multi-property-interleavings). Aborts on a non-positive value.
  size_t barren_interleaving_budget() const;

  /// Solver time and name accumulated across every interleaving of the run.
  /// Atomic because multi_property_check writes from parallel job threads.
  struct SolverStats
  {
    std::atomic<double> total_time_ms = 0.0;
    std::string name;
    std::once_flag name_flag;
  };

  SolverStats solver_stats;

  /// Counterexample file numbering. Run-scoped, not per-interleaving: two
  /// interleavings of one run share a phase and k, so a per-interleaving
  /// counter would make them overwrite each other's artifacts.
  std::atomic<size_t> ce_counter{0};

  /// Set when the run stopped before reaching a verdict on every property --
  /// exploration cut short after a violation, or claims skipped by
  /// --multi-fail-fast. The summary counts only what was checked, so without
  /// this the report would read as a complete account of the program.
  /// Atomic because multi_property_check sets it from parallel job threads.
  std::atomic<bool> report_incomplete{false};
};

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims,
  pytest_generator &pytest_gen,
  ctest_generator &ctest_gen);

/// Record why a coverage measurement is not exhaustive; the reason is listed
/// under the closing COVERAGE ANALYSIS INCOMPLETE line.
void note_cov_incomplete(const std::string &reason);

/// Closing line of a coverage run: whether every goal was actually decided,
/// and if not, why. Printed by the driver once, after the last [Coverage]
/// block (k-induction prints one per phase).
void report_coverage_completeness();

#endif
