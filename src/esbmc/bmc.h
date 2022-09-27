#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

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

  std::shared_ptr<smt_convt> runtime_solver;
  std::shared_ptr<reachability_treet> symex;
  virtual smt_convt::resultt run_decision_procedure(
    std::shared_ptr<smt_convt> &smt_conv,
    std::shared_ptr<symex_target_equationt> &eq);

  virtual void do_cbmc(
    std::shared_ptr<smt_convt> &smt_conv,
    std::shared_ptr<symex_target_equationt> &eq);

  virtual void show_program(std::shared_ptr<symex_target_equationt> &eq);
  virtual void report_success();
  virtual void report_failure();

  virtual void error_trace(
    std::shared_ptr<smt_convt> &smt_conv,
    std::shared_ptr<symex_target_equationt> &eq);

  virtual void successful_trace();

  virtual void show_vcc(std::shared_ptr<symex_target_equationt> &eq);

  virtual void
  show_vcc(std::ostream &out, std::shared_ptr<symex_target_equationt> &eq);

  virtual void report_trace(
    smt_convt::resultt &res,
    std::shared_ptr<symex_target_equationt> &eq);

  virtual void report_multi_property_trace(
        smt_convt::resultt &res,
        std::shared_ptr<symex_target_equationt> &eq,
        const std::string &msg);

  virtual void report_result(smt_convt::resultt &res);

  virtual void bidirectional_search(
    std::shared_ptr<smt_convt> &smt_conv,
    std::shared_ptr<symex_target_equationt> &eq);

  smt_convt::resultt run_thread(std::shared_ptr<symex_target_equationt> &eq);
    smt_convt::resultt multi_property_check(std::shared_ptr<symex_target_equationt> &eq, size_t remaining_claims);
  std::vector<std::unique_ptr<ssa_step_algorithm>> algorithms;
};

#endif
