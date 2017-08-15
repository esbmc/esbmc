/*******************************************************************\

Module: Generate Equation using Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BASIC_SYMEX_EQUATION_H
#define CPROVER_BASIC_SYMEX_EQUATION_H

#include <boost/shared_ptr.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <goto-programs/goto_program.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/symex_target.h>
#include <list>
#include <map>
#include <solvers/smt/smt_conv.h>
#include <util/config.h>
#include <util/irep2.h>
#include <util/namespace.h>
#include <vector>

class symex_target_equationt:public symex_targett
{
public:
  class SSA_stept;

  symex_target_equationt(const namespacet &_ns):ns(_ns)
  {
    debug_print = config.options.get_bool_option("symex-ssa-trace");
    ssa_trace = config.options.get_bool_option("ssa-trace");
  }

  // assignment to a variable - must be symbol
  // the value is destroyed
  void assignment(
    const expr2tc &guard,
    const expr2tc &lhs,
    const expr2tc &original_lhs,
    const expr2tc &rhs,
    const sourcet &source,
    std::vector<stack_framet> stack_trace,
    assignment_typet assignment_type) override ;

  // output
  void output(
    const expr2tc &guard,
    const sourcet &source,
    const std::string &fmt,
    const std::list<expr2tc> &args) override ;

  // record an assumption
  // cond is destroyed
  void assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const sourcet &source) override ;

  // record an assertion
  // cond is destroyed
  void assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<stack_framet> stack_trace,
    const sourcet &source) override ;

  void renumber(
    const expr2tc &guard,
    const expr2tc &symbol,
    const expr2tc &size,
    const sourcet &source) override ;

  virtual void convert(smt_convt &smt_conv);
  void convert_internal_step(
    smt_convt &smt_conv,
    const smt_ast *&assumpt_ast,
    smt_convt::ast_vec &assertions,
    SSA_stept &s);

  class SSA_stept
  {
  public:
    sourcet source;
    goto_trace_stept::typet type;

    // One stack trace recorded per function activation record. Valid for
    // assignment and assert steps only. In reverse order (most recent in idx
    // 0).
    std::vector<stack_framet> stack_trace;
    
    bool is_assert() const     { return type==goto_trace_stept::ASSERT; }
    bool is_assume() const     { return type==goto_trace_stept::ASSUME; }
    bool is_assignment() const { return type==goto_trace_stept::ASSIGNMENT; }
    bool is_output() const     { return type==goto_trace_stept::OUTPUT; }
    bool is_renumber() const   { return type==goto_trace_stept::RENUMBER; }
    bool is_skip() const   { return type==goto_trace_stept::SKIP; }
    
    expr2tc guard;

    // for ASSIGNMENT
    expr2tc lhs, rhs, original_lhs;
    assignment_typet assignment_type;

    // for ASSUME/ASSERT
    expr2tc cond;
    std::string comment;

    // for OUTPUT
    std::string format_string;
    std::list<expr2tc> output_args;

    // for conversion
    const smt_ast *guard_ast, *cond_ast;
    std::list<expr2tc> converted_output_args;

    // for slicing
    bool ignore;

    SSA_stept() : ignore(false)
    {
    }

    void output(const namespacet &ns, std::ostream &out) const;
    void short_output(const namespacet &ns, std::ostream &out,
                      bool show_ignored = false) const;
  };

  unsigned count_ignored_SSA_steps() const
  {
    unsigned i=0;
    for(const auto & SSA_step : SSA_steps)
      if(SSA_step.ignore) i++;
    return i;
  }

  typedef std::list<SSA_stept> SSA_stepst;
  SSA_stepst SSA_steps;

  SSA_stepst::iterator get_SSA_step(unsigned s)
  {
    SSA_stepst::iterator it=SSA_steps.begin();
    for(; s!=0; s--)
    {
      assert(it!=SSA_steps.end());
      it++;
    }
    return it;
  }

  void output(std::ostream &out) const;
  void short_output(std::ostream &out,
                    bool show_ignored = false) const;

  void check_for_duplicate_assigns() const;

  void clear()
  {
    SSA_steps.clear();
  }

  unsigned int clear_assertions();

  boost::shared_ptr<symex_targett> clone() const override 
  {
    // No pointers or anything that requires ownership modification, can just
    // duplicate self.
    return boost::shared_ptr<symex_targett>(new symex_target_equationt(*this));
  }

  void push_ctx() override ;
  void pop_ctx() override ;

protected:
  const namespacet &ns;
  bool debug_print;
  bool ssa_trace;
};

class runtime_encoded_equationt : public symex_target_equationt
{
public:
  class dual_unsat_exception { };

  runtime_encoded_equationt(const namespacet &_ns, smt_convt &conv);

  void push_ctx() override ;
  void pop_ctx() override ;

  boost::shared_ptr<symex_targett> clone() const override ;

  void convert(smt_convt &smt_conv) override ;
  void flush_latest_instructions();

  tvt ask_solver_question(const expr2tc &question);

  smt_convt &conv;
  std::list<smt_convt::ast_vec> assert_vec_list;
  std::list<const smt_ast *> assumpt_chain;
  std::list<SSA_stepst::iterator> scoped_end_points;
  SSA_stepst::iterator cvt_progress;
};

extern inline bool operator<(
  const symex_target_equationt::SSA_stepst::const_iterator a,
  const symex_target_equationt::SSA_stepst::const_iterator b)
{
  return &(*a)<&(*b);
}

std::ostream &operator<<(std::ostream &out, const symex_target_equationt::SSA_stept &step);
std::ostream &operator<<(std::ostream &out, const symex_target_equationt &equation);

#endif
