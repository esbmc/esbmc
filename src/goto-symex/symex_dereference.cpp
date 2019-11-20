/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <langapi/language_util.h>
#include <pointer-analysis/dereference.h>
#include <util/irep2.h>
#include <util/migrate.h>

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
  if(is_symbol2t(expr))
  {
    // Null and invalid name lookups will fail.
    if(
      to_symbol2t(expr).thename == "NULL" ||
      to_symbol2t(expr).thename == "INVALID")
      return false;

    const symbolt &ptr_symbol = goto_symex.ns.lookup(to_symbol2t(expr).thename);

    const irep_idt &failed_symbol = ptr_symbol.type.failed_symbol();

    if(failed_symbol == "")
      return false;

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
}

void symex_dereference_statet::dump_internal_state(
  const std::list<struct internal_item> &data)
{
  goto_symex.internal_deref_items.insert(
    goto_symex.internal_deref_items.begin(), data.begin(), data.end());
}

bool symex_dereference_statet::is_live_variable(const expr2tc &symbol)
{
  expr2tc sym = symbol;

  // NB, symbols shouldn't hit this point with no renaming (i.e. level0),
  // this should eventually be asserted.
  if(
    to_symbol2t(sym).rlevel == symbol2t::level0 ||
    to_symbol2t(sym).rlevel == symbol2t::level1_global ||
    to_symbol2t(sym).rlevel == symbol2t::level2_global)
    return true;

  goto_symex.replace_dynamic_allocation(sym);
  goto_symex.replace_nondet(sym);
  goto_symex.dereference(sym, dereferencet::INTERNAL);

  // Symbol is renamed to at least level 1, fetch the relevant thread data
  const execution_statet &ex_state = goto_symex.art1->get_cur_state();
  const goto_symex_statet &state =
    ex_state.threads_state[to_symbol2t(sym).thread_num];

  // Level one names represent the storage for a variable, and this symbol
  // may have entered pointer tracking at any time the variable had its address
  // taken (and subsequently been propagated from there). If the stack frame
  // that that variable was in has now expired, it's an invalid pointer. Look
  // up the stack frames currently active the corresponding thread to see
  // whether there are any records for the lexical variable that have this
  // activation record.

  for(auto it = state.call_stack.rbegin(); it != state.call_stack.rend(); it++)
  {
    auto &local_vars = it->local_variables;
    if(
      local_vars.find(renaming::level2t::name_record(to_symbol2t(sym))) !=
      local_vars.end())
      return true;
  }

  // There were no stack frames where that variable existed and had the correct
  // level1 num: it's dead Jim.
  return false;
}

void goto_symext::dereference(expr2tc &expr, dereferencet::modet mode)
{
  symex_dereference_statet symex_dereference_state(*this, *cur_state);

  dereferencet dereference(ns, new_context, options, symex_dereference_state);

  // needs to be renamed to level 1
  assert(!cur_state->call_stack.empty());
  cur_state->top().level1.rename(expr);

  guardt guard;
  switch(mode)
  {
  case dereferencet::FREE:
  {
    expr2tc tmp = expr;
    while(is_typecast2t(tmp))
      tmp = to_typecast2t(tmp).from;

    assert(is_pointer_type(tmp));
    std::list<expr2tc> dummy;
    // Dereference to byte type, because it's guarenteed to succeed.
    tmp = dereference2tc(get_uint8_type(), tmp);

    dereference.dereference_expr(tmp, guard, dereferencet::FREE);
    expr = tmp;
    break;
  }

  default:
    dereference.dereference_expr(expr, guard, mode);
  }
}
