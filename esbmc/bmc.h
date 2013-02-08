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
    ui(ui_message_handlert::PLAIN)
  {
    interleaving_number = 0;
    interleaving_failed = 0;
    uw_loop = 0;

    const symbolt *sp;
    if (ns.lookup(irep_idt("c::__ESBMC_alloc"), sp))
      is_cpp = true;
    else
      is_cpp = false;


#ifdef Z3
    runtime_z3_conv = new z3_convt(opts.get_bool_option("int-encoding"),
                                   opts.get_bool_option("smt"), is_cpp, ns);

    runtime_z3_conv->set_filename(opts.get_option("outfile"));
    runtime_z3_conv->set_z3_core_size(
                                    atol(opts.get_option("core-size").c_str()));

    if (options.get_bool_option("smt-during-symex")) {
      symex = new reachability_treet(funcs, ns, options,
                          new runtime_encoded_equationt(ns, *runtime_z3_conv),
                          _context);
    } else {
#endif
      symex = new reachability_treet(funcs, ns, options,
                                     new symex_target_equationt(ns),
                                     _context);
#ifdef Z3
    }
#endif
  }

  optionst &options;

  unsigned int interleaving_number;
  unsigned int interleaving_failed;
  unsigned int uw_loop;
  bool is_cpp;

  virtual bool run(void);
  virtual ~bmct() { }

  void set_ui(language_uit::uit _ui) { ui=_ui; }

protected:
  const contextt &context;
  namespacet ns;
#ifdef Z3
  z3_convt *runtime_z3_conv;
#endif
  reachability_treet *symex;

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
    z3_solver(bmct &bmc, bool is_cpp, const namespacet &ns);
    virtual bool run_solver(symex_target_equationt &equation);
  protected:
    z3_convt z3_conv;
  };

  class z3_runtime_solver : public solver_base {
  public:
    z3_runtime_solver(bmct &bmc, bool is_cpp, z3_convt *conv);
    virtual bool run_solver(symex_target_equationt &equation);
  protected:
    z3_convt *z3_conv;
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

  virtual prop_convt::resultt
    run_decision_procedure(prop_convt &prop_conv,
                           symex_target_equationt &equation);

  virtual void do_cbmc(prop_convt &solver, symex_target_equationt &eq);
  virtual void show_vcc(symex_target_equationt &equation);
  virtual void show_vcc(std::ostream &out, symex_target_equationt &equation);
  virtual void show_program(symex_target_equationt &equation);
  virtual void report_success();
  virtual void report_failure();
  virtual void write_checkpoint();

  virtual void error_trace(
    prop_convt &prop_conv, symex_target_equationt &equation);
    bool run_thread();
};

#endif
