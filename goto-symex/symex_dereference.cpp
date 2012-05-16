/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <pointer-analysis/dereference.h>
#include <langapi/language_util.h>

#include "goto_symex.h"
#include "renaming_ns.h"

class symex_dereference_statet:
  public dereference_callbackt
{
public:
  symex_dereference_statet(
    goto_symext &_goto_symex,
    goto_symext::statet &_state):
    goto_symex(_goto_symex),
    state(_state)
  {
  }

protected:
  goto_symext &goto_symex;
  goto_symext::statet &state;

  // overloads from dereference_callbackt
  // XXXjmorse - no it doesn't. This should be virtual pure!
  virtual bool is_valid_object(const irep_idt &identifier __attribute__((unused)))
  {
    return true;
  }
#if 1
  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard);
#endif
  virtual void get_value_set(
    const expr2tc &expr,
    value_setst::valuest &value_set);

  virtual bool has_failed_symbol(
    const expr2tc &expr,
    const symbolt *&symbol);
};

void symex_dereference_statet::dereference_failure(
  const std::string &property __attribute__((unused)),
  const std::string &msg __attribute__((unused)),
  const guardt &guard __attribute__((unused)))
{
  // XXXjmorse - this is clearly wrong, but we can't do anything about it until
  // we fix the memory model.
}

bool symex_dereference_statet::has_failed_symbol(
  const expr2tc &expr,
  const symbolt *&symbol)
{
  renaming_nst renaming_ns(goto_symex.ns, state);

  if (is_symbol2t(expr))
  {
    const symbolt &ptr_symbol = renaming_ns.lookup(to_symbol2t(expr).name);

    const irep_idt &failed_symbol=
      ptr_symbol.type.failed_symbol();

    if(failed_symbol=="") return false;

    return !renaming_ns.lookup(failed_symbol, symbol);
  }

  return false;
}

void symex_dereference_statet::get_value_set(
  const expr2tc &expr,
  value_setst::valuest &value_set)
{
  renaming_nst renaming_ns(goto_symex.ns, state);

  state.value_set.get_value_set(expr, value_set, renaming_ns);
}

void goto_symext::dereference_rec(
  expr2tc &expr,
  guardt &guard,
  dereferencet &dereference,
  const bool write)
{

  if (is_dereference2t(expr))
  {
    dereference2t &deref = to_dereference2t(expr);

    // first make sure there are no dereferences in there
    dereference_rec(deref.value, guard, dereference, false);

    dereference.dereference(deref.value, guard,
                            write ? dereferencet::WRITE : dereferencet::READ);
    expr = deref.value;
  }
  else if (is_index2t(expr) &&
           is_pointer_type(to_index2t(expr).source_value->type))
  {
    index2t &index = to_index2t(expr);
    expr2tc tmp = expr2tc(new add2t(index.source_value->type,
                                    index.source_value, index.index));

    // first make sure there are no dereferences in there
    dereference_rec(tmp, guard, dereference, false);

    dereference.dereference(tmp, guard,
                            write ? dereferencet::WRITE : dereferencet::READ);
    expr = tmp;
  }
  else
  {
    std::vector<expr2tc *> operands;
    expr.get()->list_operands(operands);
    for (std::vector<expr2tc *>::const_iterator it = operands.begin();
         it != operands.end(); it++)
      dereference_rec(**it, guard, dereference, write);
  }
}

void goto_symext::dereference(expr2tc &expr, const bool write)
{
  symex_dereference_statet symex_dereference_state(*this, *cur_state);
  renaming_nst renaming_ns(ns, *cur_state);

  dereferencet dereference(
    renaming_ns,
    new_context,
    options,
    symex_dereference_state);

  // needs to be renamed to level 1
  assert(!cur_state->call_stack.empty());
  cur_state->top().level1.rename(expr);

  guardt guard;
  dereference_rec(expr, guard, dereference, write);
}
