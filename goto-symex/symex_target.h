/*******************************************************************\

Module: Generate Equation using Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SYMEX_TARGET_H
#define CPROVER_GOTO_SYMEX_SYMEX_TARGET_H

#include <irep2.h>
#include <vector>
#include <expr.h>
#include <symbol.h>
#include <guard.h>

#include <goto-programs/goto_program.h>

class symex_targett
{
public:
  virtual ~symex_targett() { }

  struct sourcet
  {
    unsigned thread_nr;
    goto_programt::const_targett pc;
    const goto_programt *prog;
    bool is_set;

    sourcet():thread_nr(0), prog(NULL), is_set(false)
    {
    }

    sourcet(goto_programt::const_targett _pc, const goto_programt *_prog):
      thread_nr(0), pc(_pc), prog(_prog), is_set(true)
    {
      is_set=true;
    }
  };

  typedef enum { STATE, HIDDEN } assignment_typet;

  // write to a variable - must be symbol
  // the value is destroyed
  virtual void assignment(
    const expr2tc &guard,
    const expr2tc &lhs,
    const expr2tc &original_lhs,
    const expr2tc &rhs,
    const sourcet &source,
    std::vector<dstring> stack_trace,
    assignment_typet assignment_type)=0;

  // record output
  virtual void output(
    const expr2tc &guard,
    const sourcet &source,
    const std::string &fmt,
    const std::list<expr2tc> &args)=0;

  // record an assumption
  // cond is destroyed
  virtual void assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const sourcet &source)=0;

  // record an assertion
  // cond is destroyed
  virtual void assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<dstring> stack_trace,
    const sourcet &source)=0;

  // Renumber the pointer object of a given symbol
  virtual void renumber(
    const expr2tc &guard,
    const expr2tc &symbol,
    const expr2tc &size,
    const sourcet &source)=0;

  // Abstract method, with the purpose of duplicating a symex_targett from the
  // subclass.
  virtual symex_targett *clone(void) const = 0;

  virtual void push_ctx(void) = 0;
  virtual void pop_ctx(void) = 0;
};

bool operator < (
  const symex_targett::sourcet &a,
  const symex_targett::sourcet &b);

#endif
