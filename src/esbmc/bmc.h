/*******************************************************************\

Module: Bounded Model Checking for ANSI-C + HDL

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

#include <boost/shared_ptr.hpp>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <langapi/language_ui.h>
#include <list>
#include <map>
#include <solvers/smt/smt_conv.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <util/hash_cont.h>
#include <util/options.h>

class bmct:public messaget
{
public:
  bmct(
    const goto_functionst &funcs,
    optionst &opts,
    contextt &_context,
    message_handlert &_message_handler);

  optionst &options;
  enum {
    ltl_res_good,
    ltl_res_succeeding,
    ltl_res_failing,
    ltl_res_bad
  };
  int ltl_results_seen[4];

  BigInt interleaving_number;
  unsigned int interleaving_failed;

  virtual smt_convt::resultt start_bmc();
  virtual smt_convt::resultt run(boost::shared_ptr<symex_target_equationt> &eq);
  virtual ~bmct() { }

  void set_ui(language_uit::uit _ui) { ui=_ui; }

protected:
  const contextt &context;
  namespacet ns;
  boost::shared_ptr<smt_convt> runtime_solver;
  std::shared_ptr<reachability_treet> symex;

  // use gui format
  language_uit::uit ui;

  virtual smt_convt::resultt run_decision_procedure(
    boost::shared_ptr<smt_convt> &smt_conv,
    boost::shared_ptr<symex_target_equationt> &eq);

  virtual void do_cbmc(
    boost::shared_ptr<smt_convt> &smt_conv,
    boost::shared_ptr<symex_target_equationt> &eq);

  virtual void show_program(boost::shared_ptr<symex_target_equationt> &eq);
  virtual void report_success();
  virtual void report_failure();

  virtual void error_trace(
    boost::shared_ptr<smt_convt> &smt_conv,
    boost::shared_ptr<symex_target_equationt> &eq);

  virtual void successful_trace(boost::shared_ptr<symex_target_equationt> &eq);

  virtual void show_vcc(boost::shared_ptr<symex_target_equationt> &eq);

  virtual void show_vcc(
    std::ostream &out,
    boost::shared_ptr<symex_target_equationt> &eq);

  smt_convt::resultt run_thread(boost::shared_ptr<symex_target_equationt> &eq);
  int ltl_run_thread(boost::shared_ptr<symex_target_equationt> &eq);
};

#endif
