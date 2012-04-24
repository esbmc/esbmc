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

#include <solvers/prop/prop.h>
#include <solvers/prop/prop_conv.h>
#ifdef Z3
#include <solvers/z3/z3_conv.h>
#endif
#include <langapi/language_ui.h>
#include <goto-symex/symex_target_equation.h>
#include <goto-symex/reachability_tree.h>

class bmct:public messaget
{
public:
  bmct(const goto_functionst &funcs, optionst &opts,
       contextt &_context, message_handlert &_message_handler):
    messaget(_message_handler),
    options(opts),
    context(_context),
    ns(_context),
    symex(funcs, ns, options, new symex_target_equationt(ns), _context),
    ui(ui_message_handlert::PLAIN)
  {
    _unsat_core=0;
    interleaving_number = 0;
    interleaving_failed = 0;
    uw_loop = 0;

    const symbolt *sp;
    if (ns.lookup(irep_idt("c::__ESBMC_alloc"), sp))
      is_cpp = true;
    else
      is_cpp = false;
  }

  uint _unsat_core;
  uint _number_of_assumptions;
  optionst &options;

  unsigned int interleaving_number;
  unsigned int interleaving_failed;
  unsigned int uw_loop;
  bool is_cpp;

  virtual bool run(const goto_functionst &goto_functions);
  virtual ~bmct() { }

  // additional stuff
  expr_listt bmc_constraints;

  void set_ui(language_uit::uit _ui) { ui=_ui; }

protected:
  const contextt &context;
  namespacet ns;
  reachability_treet symex;

  // use gui format
  language_uit::uit ui;

  class solver_base {
  public:
    virtual bool run_solver(symex_target_equationt &equation);
    virtual ~solver_base() {}

  protected:
    solver_base(bmct &_bmc) : bmc(_bmc)
    { }

    prop_convt *conv;
    bmct &bmc;
  };

#ifdef Z3
  class z3_solver : public solver_base {
  public:
    z3_solver(bmct &bmc, bool is_cpp);
    virtual bool run_solver(symex_target_equationt &equation);
  protected:
    z3_convt z3_conv;
  };
#endif

  class output_solver : public solver_base {
  public:
    output_solver(bmct &bmc);
    ~output_solver();
    virtual bool run_solver(symex_target_equationt &equation);
  protected:
    virtual bool write_output() = 0;
    std::ostream *out_file;
  };

  virtual decision_proceduret::resultt
    run_decision_procedure(prop_convt &prop_conv,
                           symex_target_equationt &equation);

  virtual void do_unwind_module(
    decision_proceduret &decision_procedure);

  virtual void do_cbmc(prop_convt &solver, symex_target_equationt &eq);
  virtual void show_vcc(symex_target_equationt &equation);
  virtual void show_vcc(std::ostream &out, symex_target_equationt &equation);
  virtual void show_program(symex_target_equationt &equation);
  virtual void report_success();
  virtual void report_failure();
  virtual void write_checkpoint();

  virtual void error_trace(
    const prop_convt &prop_conv, symex_target_equationt &equation);
    bool run_thread();
};

#endif
