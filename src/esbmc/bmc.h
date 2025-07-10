#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

#include <goto-programs/goto_coverage.h>
#include <goto-symex/slice.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <langapi/language_ui.h>
#include <list>
#include <map>
#include <solvers/smt/smt_conv.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <util/options.h>
#include <util/algorithms.h>
#include <util/cmdline.h>
#include <atomic>

class bmct
{
public:
  bmct(goto_functionst &funcs, optionst &opts, contextt &_context);

  optionst &options;
  enum ltl_res
  {
    ltl_res_good,
    ltl_res_succeeding,
    ltl_res_failing,
    ltl_res_bad
  };
  size_t ltl_results_seen[4];

  BigInt interleaving_number;
  BigInt interleaving_failed;

  virtual smt_convt::resultt start_bmc();
  virtual smt_convt::resultt run(std::shared_ptr<symex_target_equationt> &eq);
  virtual ~bmct() = default;

protected:
  const contextt &context;
  namespacet ns;

  std::unique_ptr<smt_convt> runtime_solver;
  std::unique_ptr<reachability_treet> symex;
  mutable std::atomic<bool> keep_alive_running;
  mutable std::atomic<int> keep_alive_interval;

  virtual smt_convt::resultt
  run_decision_procedure(smt_convt &smt_conv, symex_target_equationt &eq) const;

  virtual void show_program(const symex_target_equationt &eq);
  virtual void report_success();
  virtual void report_failure();
  virtual void keep_alive_function() const;

  virtual void
  error_trace(smt_convt &smt_conv, const symex_target_equationt &eq);

  virtual void successful_trace();

  virtual void show_vcc(const symex_target_equationt &eq);

  virtual void show_vcc(std::ostream &out, const symex_target_equationt &eq);

  virtual void
  report_trace(smt_convt::resultt &res, const symex_target_equationt &eq);

  virtual void report_result(smt_convt::resultt &res);

  virtual void
  bidirectional_search(smt_convt &smt_conv, const symex_target_equationt &eq);

  smt_convt::resultt run_thread(std::shared_ptr<symex_target_equationt> &eq);

  int ltl_run_thread(symex_target_equationt &equation) const;

  smt_convt::resultt multi_property_check(
    const symex_target_equationt &eq,
    size_t remaining_claims);

  std::vector<std::unique_ptr<ssa_step_algorithm>> algorithms;

  void generate_smt_from_equation(
    smt_convt &smt_conv,
    symex_target_equationt &eq) const;

  // for multi-property
  void
  clear_verified_claims(const claim_slicer &claim, const bool &is_goto_cov);

  virtual void report_multi_property_trace(
    const smt_convt::resultt &res,
    const std::unique_ptr<smt_convt> &solver,
    const symex_target_equationt &local_eq,
    const std::atomic<size_t> ce_counter,
    const goto_tracet &goto_trace,
    const std::string &msg);

  void report_coverage_verbose(
    const claim_slicer &claim,
    const std::string &claim_sig,
    const bool &is_assert_cov,
    const bool &is_cond_cov,
    const bool &is_branch_cov,
    const bool &is_branch_func_cov,
    const std::unordered_set<std::string> &reached_claims,
    const std::unordered_multiset<std::string> &reached_mul_claims);
};

void report_coverage(
  const optionst &options,
  std::unordered_set<std::string> &reached_claims,
  const std::unordered_multiset<std::string> &reached_mul_claims);

#endif
