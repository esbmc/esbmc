/*******************************************************************\

Module: Generate Equation using Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BASIC_SYMEX_EQUATION_H
#define CPROVER_BASIC_SYMEX_EQUATION_H

#include <irep2.h>

extern "C" {
#include <stdio.h>
}

#include <list>
#include <map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <namespace.h>

#include <config.h>
#include <goto-programs/goto_program.h>
#include <solvers/smt/smt_conv.h>

#include "symex_target.h"
#include "goto_trace.h"

extern "C" {
#include <stdint.h>
#include <string.h>
}

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
  virtual void assignment(
    const expr2tc &guard,
    const expr2tc &lhs,
    const expr2tc &original_lhs,
    const expr2tc &rhs,
    const sourcet &source,
    std::vector<stack_framet> stack_trace,
    assignment_typet assignment_type);

  // output
  virtual void output(
    const expr2tc &guard,
    const sourcet &source,
    const std::string &fmt,
    const std::list<expr2tc> &args);

  // record an assumption
  // cond is destroyed
  virtual void assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const sourcet &source);

  // record an assertion
  // cond is destroyed
  virtual void assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<stack_framet> stack_trace,
    const sourcet &source);

  virtual void renumber(
    const expr2tc &guard,
    const expr2tc &symbol,
    const expr2tc &size,
    const sourcet &source);

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
    for(SSA_stepst::const_iterator
        it=SSA_steps.begin();
        it!=SSA_steps.end(); it++)
      if(it->ignore) i++;
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

  virtual boost::shared_ptr<symex_targett> clone(void) const
  {
    // No pointers or anything that requires ownership modification, can just
    // duplicate self.
    return boost::shared_ptr<symex_targett>(new symex_target_equationt(*this));
  }

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

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

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  virtual boost::shared_ptr<symex_targett> clone(void) const;

  virtual void convert(smt_convt &smt_conv);
  void flush_latest_instructions(void);

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
