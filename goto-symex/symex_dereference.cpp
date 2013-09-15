/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <pointer-analysis/dereference.h>
#include <langapi/language_util.h>

#include "goto_symex.h"

static inline bool is_non_scalar_expr(const expr2tc &e)
{
  return is_member2t(e) || is_index2t(e) || (is_if2t(e) && !is_scalar_type(e));
}

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

void goto_symext::dereference_rec(
  expr2tc &expr,
  guardt &guard,
  dereferencet &dereference,
  const bool write)
{

  if (is_dereference2t(expr))
  {
    assert((is_scalar_type(expr) || is_code_type(expr))
           && "Can't dereference to a nonscalar type");

    dereference2t &deref = to_dereference2t(expr);

    // first make sure there are no dereferences in there
    dereference_rec(deref.value, guard, dereference, false);

    expr2tc result = dereference.dereference(deref.value, guard,
                            write ? dereferencet::WRITE : dereferencet::READ);
    expr = result;
  }
  else if (is_index2t(expr) &&
           is_pointer_type(to_index2t(expr).source_value))
  {
    assert((is_scalar_type(expr) || is_code_type(expr))
           && "Can't dereference to a nonscalar type");
    index2t &index = to_index2t(expr);
    add2tc tmp(index.source_value->type, index.source_value, index.index);

    // first make sure there are no dereferences in there
    dereference_rec(tmp, guard, dereference, false);

    expr2tc result = dereference.dereference(tmp, guard,
                            write ? dereferencet::WRITE : dereferencet::READ);
    expr = result;
  }
  else if (is_non_scalar_expr(expr))
  {
    // The result of this expression should be scalar: we're transitioning
    // from a scalar result to a nonscalar result.
    // Unless we're doing something crazy with multidimensional arrays and
    // address_of, for example, where no dereference is involved. In that case,
    // bail.
    bool contains_deref = dereference.has_dereference(expr);
    if (!contains_deref)
      return;

    assert(is_scalar_type(expr));

    expr2tc res =
      dereference_rec_nonscalar(expr, expr, guard, dereference, write);

    // If a dereference successfully occurred, replace expr at this level.
    // XXX -- explain this better.
    if (!is_nil_expr(res))
      expr = res;
  }
  else
  {
    Forall_operands2(it, idx, expr) {
      if (is_nil_expr(*it))
        continue;

      dereference_rec(*it, guard, dereference, write);
    }
  }
}

expr2tc
goto_symext::dereference_rec_nonscalar(
  expr2tc &expr,
  const expr2tc &top_scalar,
  guardt &guard,
  dereferencet &dereference,
  const bool write)
{

  if (is_dereference2t(expr))
  {
    dereference2t &deref = to_dereference2t(expr);
    // first make sure there are no dereferences in there
    dereference_rec(deref.value, guard, dereference, false);
    expr2tc result = dereference.dereference(deref.value, guard,
                            write ? dereferencet::WRITE : dereferencet::READ,
                            top_scalar);
    return result;
  }
  else if (is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value))
  {
    index2t &index = to_index2t(expr);
    add2tc tmp(index.source_value->type, index.source_value, index.index);

    // first make sure there are no dereferences in there
    dereference_rec(tmp, guard, dereference, false);

    expr2tc result = dereference.dereference(tmp, guard,
                            write ? dereferencet::WRITE : dereferencet::READ,
                            top_scalar);
    return result;
  }
  else if (is_non_scalar_expr(expr))
  {
    if (is_member2t(expr)) {
      return dereference_rec_nonscalar(to_member2t(expr).source_value,
                                       top_scalar, guard, dereference, write);
    } else if (is_index2t(expr)) {
      dereference_rec(to_index2t(expr).index, guard, dereference, write);
      return dereference_rec_nonscalar(to_index2t(expr).source_value,
                                       top_scalar, guard, dereference, write);
    } else if (is_if2t(expr)) {
      guardt g1 = guard, g2 = guard;
      if2t &theif = to_if2t(expr);
      g1.add(theif.cond);
      g2.add(not2tc(theif.cond));
      expr2tc res1 = dereference_rec_nonscalar(theif.true_value, top_scalar, g1,
                                               dereference, write);
      expr2tc res2 = dereference_rec_nonscalar(theif.false_value, top_scalar,
                                               g1, dereference, write);
      if2tc fin(res1->type, theif.cond, res1, res2);
      return fin;
    } else {
      std::cerr << "Unexpected expression in dereference_rec_nonscalar"
                << std::endl;
      expr->dump();
      abort();
    }
  }
  else if (is_typecast2t(expr))
  {
    // Just blast straight through
    return dereference_rec_nonscalar(to_typecast2t(expr).from, top_scalar,
                                     guard, dereference, write);
  }
  else
  {
    // This should end up being either a constant or a symbol; either way
    // there should be no sudden transition back to scalars, except through
    // dereferences. Return nil to indicate that there was no dereference at
    // the bottom of this.
    assert(!is_scalar_type(expr) &&
           (is_constant_expr(expr) || is_symbol2t(expr)));
    return expr2tc();
  }
}

void goto_symext::dereference(expr2tc &expr, const bool write)
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
  dereference_rec(expr, guard, dereference, write);
}
