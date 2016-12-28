/*******************************************************************\

Module: Bounded Model Checking for ANSI-C + HDL

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_BMC_H
#define CPROVER_CBMC_BMC_H

#include <list>
#include <map>

#include <hash_cont.h>
#include <options.h>

#include <solvers/solve.h>
#include <solvers/smt/smt_conv.h>
#ifdef Z3
#include <solvers/z3/z3_conv.h>
#endif
#include <solvers/smtlib/smtlib_conv.h>
#include <langapi/language_ui.h>
#include <goto-symex/symex_target_equation.h>
#include <goto-symex/reachability_tree.h>

class bmct:public messaget
{
public:
  bmct(const goto_functionst &funcs, optionst &opts,
      contextt &_context, message_handlert &_message_handler) :
    messaget(_message_handler),
    options(opts),
    context(_context),
    ns(_context),
    ui(ui_message_handlert::PLAIN)
  {
    interleaving_number = 0;
    interleaving_failed = 0;
    uw_loop = 0;

    ltl_results_seen[ltl_res_bad] = 0;
    ltl_results_seen[ltl_res_failing] = 0;
    ltl_results_seen[ltl_res_succeeding] = 0;
    ltl_results_seen[ltl_res_good] = 0;

    if (options.get_bool_option("smt-during-symex")) {
      runtime_solver = create_solver_factory(
        "", opts.get_bool_option("int-encoding"), ns, options);

      symex =
        new reachability_treet(
          funcs,
          ns,
          options,
          std::shared_ptr<runtime_encoded_equationt>(
            new runtime_encoded_equationt(ns, *runtime_solver)),
          _context,
          _message_handler);
    } else {
      symex =
        new reachability_treet(
          funcs,
          ns,
          options,
          std::shared_ptr<symex_target_equationt>(
            new symex_target_equationt(ns)),
          _context,
          _message_handler);
    }
  }

  optionst &options;
  enum {
    ltl_res_good,
    ltl_res_succeeding,
    ltl_res_failing,
    ltl_res_bad
  };
  int ltl_results_seen[4];

  unsigned int interleaving_number;
  unsigned int interleaving_failed;
  unsigned int uw_loop;

  virtual bool run(void);
  virtual ~bmct() { }

  void set_ui(language_uit::uit _ui) { ui=_ui; }

protected:
  const contextt &context;
  namespacet ns;
  smt_convt *runtime_solver;
  reachability_treet *symex;

  // use gui format
  language_uit::uit ui;

  virtual smt_convt::resultt
    run_decision_procedure(smt_convt &smt_conv,
                           symex_target_equationt &equation);

  virtual void do_cbmc(smt_convt &solver, symex_target_equationt &eq);
  virtual bool run_solver(symex_target_equationt &equation, smt_convt *solver);
  virtual void show_vcc(symex_target_equationt &equation);
  virtual void show_vcc(std::ostream &out, symex_target_equationt &equation);
  virtual void show_program(symex_target_equationt &equation);
  virtual void report_success();
  virtual void report_failure();
  virtual void write_checkpoint();

  virtual void error_trace(
    smt_convt &smt_conv, symex_target_equationt &equation);
  virtual void successful_trace(symex_target_equationt &equation);
    bool run_thread();
    int ltl_run_thread(symex_target_equationt *equation);
};

#endif
