/*******************************************************************\

Module: Generate Equation using Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SYMEX_TARGET_H
#define CPROVER_GOTO_SYMEX_SYMEX_TARGET_H

#include <boost/shared_ptr.hpp>
#include <goto-programs/goto_program.h>
#include <util/expr.h>
#include <util/guard.h>
#include <util/irep2.h>
#include <util/symbol.h>
#include <vector>

class stack_framet;

class symex_targett
{
public:
  virtual ~symex_targett() = default;

  struct sourcet
  {
    unsigned thread_nr;
    goto_programt::const_targett pc;
    const goto_programt *prog;
    bool is_set;

    sourcet() : thread_nr(0), prog(nullptr), is_set(false)
    {
    }

    sourcet(goto_programt::const_targett _pc, const goto_programt *_prog)
      : thread_nr(0), pc(_pc), prog(_prog), is_set(true)
    {
      is_set = true;
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
    std::vector<stack_framet> stack_trace,
    assignment_typet assignment_type) = 0;

  // record output
  virtual void output(
    const expr2tc &guard,
    const sourcet &source,
    const std::string &fmt,
    const std::list<expr2tc> &args) = 0;

  // record an assumption
  // cond is destroyed
  virtual void assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const sourcet &source) = 0;

  // record an assertion
  // cond is destroyed
  virtual void assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<stack_framet> stack_trace,
    const sourcet &source) = 0;

  // Renumber the pointer object of a given symbol
  virtual void renumber(
    const expr2tc &guard,
    const expr2tc &symbol,
    const expr2tc &size,
    const sourcet &source) = 0;

  // Abstract method, with the purpose of duplicating a symex_targett from the
  // subclass.
  virtual boost::shared_ptr<symex_targett> clone() const = 0;

  virtual void push_ctx() = 0;
  virtual void pop_ctx() = 0;
};

class stack_framet
{
public:
  stack_framet(const irep_idt &func, const symex_targett::sourcet &__src)
    : function(func), _src(__src), src(&_src)
  {
  }
  stack_framet(const irep_idt &func) : function(func), src(nullptr)
  {
  }
  stack_framet(const stack_framet &ref)
  {
    *this = ref;
    if(src != nullptr)
      src = &_src;
  }

  bool _cmp(const stack_framet &ref) const
  {
    if(function != ref.function)
      return false;
    if(src == nullptr && ref.src == src)
      return true;
    if(src == nullptr || ref.src == nullptr)
      return false;

    return src->pc->location_number == ref.src->pc->location_number;
  }

  irep_idt function;
  symex_targett::sourcet _src;
  const symex_targett::sourcet *src;
};

bool operator<(
  const symex_targett::sourcet &a,
  const symex_targett::sourcet &b);

// Can't remember how to get address of operator== defined inside class
inline bool operator==(const stack_framet &a, const stack_framet &b)
{
  return a._cmp(b);
}

#endif
