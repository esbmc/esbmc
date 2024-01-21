#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

#include <goto-programs/goto_coverage.h>
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

class bmct
{
public:
  bmct(goto_functionst &funcs, optionst &opts, contextt &_context);

  optionst &options;

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

  virtual smt_convt::resultt
  run_decision_procedure(smt_convt &smt_conv, symex_target_equationt &eq);

  virtual void show_program(const symex_target_equationt &eq);
  virtual void report_success();
  virtual void report_failure();

  virtual void
  error_trace(smt_convt &smt_conv, const symex_target_equationt &eq);

  virtual void successful_trace();

  virtual void show_vcc(const symex_target_equationt &eq);

  virtual void show_vcc(std::ostream &out, const symex_target_equationt &eq);

  virtual void
  report_trace(smt_convt::resultt &res, const symex_target_equationt &eq);

  virtual void
  report_multi_property_trace(smt_convt::resultt &res, const std::string &msg);

  virtual void report_result(smt_convt::resultt &res);

  virtual void
  bidirectional_search(smt_convt &smt_conv, const symex_target_equationt &eq);

  smt_convt::resultt run_thread(std::shared_ptr<symex_target_equationt> &eq);

  smt_convt::resultt multi_property_check(
    const symex_target_equationt &eq,
    size_t remaining_claims);

  std::vector<std::unique_ptr<ssa_step_algorithm>> algorithms;

  void
  generate_smt_from_equation(smt_convt &smt_conv, symex_target_equationt &eq);
};

#endif
