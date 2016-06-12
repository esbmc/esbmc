/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <pointer-analysis/dereference.h>
#include <langapi/language_util.h>

#include "goto_symex.h"

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
  virtual bool is_valid_object(const irep_idt &identifier __attribute__((unused)))
  {
    return true;
  }

  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard);

  virtual void get_value_set(
    const expr2tc &expr,
    value_setst::valuest &value_set);

  virtual bool has_failed_symbol(
    const expr2tc &expr,
    const symbolt *&symbol);

  virtual void rename(expr2tc &expr);

  virtual void dump_internal_state(const std::list<struct internal_item> &data);
};

void symex_dereference_statet::dereference_failure(
  const std::string &property __attribute__((unused)),
  const std::string &msg,
  const guardt &guard)
{
  expr2tc g = guard.as_expr();
  goto_symex.replace_dynamic_allocation(g);
  goto_symex.claim(not2tc(g), "dereference failure: " + msg);
}

bool symex_dereference_statet::has_failed_symbol(
  const expr2tc &expr,
  const symbolt *&symbol)
{

  if (is_symbol2t(expr))
  {
    // Null and invalid name lookups will fail.
    if (to_symbol2t(expr).thename == "NULL" ||
        to_symbol2t(expr).thename == "INVALID")
      return false;

    const symbolt &ptr_symbol = goto_symex.ns.lookup(to_symbol2t(expr).thename);

    const irep_idt &failed_symbol=
      ptr_symbol.type.failed_symbol();

    if(failed_symbol=="") return false;

    return !goto_symex.ns.lookup(failed_symbol, symbol);
  }

  return false;
}

void symex_dereference_statet::get_value_set(
  const expr2tc &expr,
  value_setst::valuest &value_set)
{

  state.value_set.get_value_set(expr, value_set);
}

void symex_dereference_statet::rename(expr2tc &expr)
{
  goto_symex.cur_state->rename(expr);
  return;
}

void
symex_dereference_statet::dump_internal_state(
                      const std::list<struct internal_item> &data)
{
  goto_symex.internal_deref_items.insert(
                          goto_symex.internal_deref_items.begin(),
                          data.begin(), data.end());
  return;
}

void goto_symext::dereference(expr2tc &expr, const bool write, bool free,
                              bool internal)
{

  symex_dereference_statet symex_dereference_state(*this, *cur_state);

  dereferencet dereference(
    ns,
    new_context,
    options,
    symex_dereference_state);

  // needs to be renamed to level 1
  assert(!cur_state->call_stack.empty());
  cur_state->top().level1.rename(expr);

  guardt guard;
  if (internal) {
    dereference.dereference_expr(expr, guard, dereferencet::INTERNAL);
  } else if (free) {
    expr2tc tmp = expr;
    while (is_typecast2t(tmp))
      tmp = to_typecast2t(tmp).from;

    assert(is_pointer_type(tmp));
    std::list<expr2tc> dummy;
    // Dereference to byte type, because it's guaranteed to succeed.
    tmp = dereference2tc(get_uint8_type(), tmp);

    dereference.dereference_expr(tmp, guard, dereferencet::FREE);
    expr = tmp;
  } else {
    dereference.dereference_expr(expr, guard, (write) ? dereferencet::WRITE
                                                      : dereferencet::READ);
  }
}
