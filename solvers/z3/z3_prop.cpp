/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>
#include <malloc.h>
#include <set>
#include <i2string.h>
#include <std_expr.h>
#include <std_types.h>

#include "z3_prop.h"
#include "z3_conv.h"

z3_propt::z3_propt(bool uw, z3_convt &_owner) : owner(_owner)
{
  _no_variables = 1;
  this->uw = uw;
}

z3_propt::~z3_propt()
{
}

literalt
z3_propt::land(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for (unsigned int i = 0; i < bv.size(); i++)
    args[i] = z3_literal(bv[i]);

  result = Z3_mk_and(z3_ctx, bv.size(), args);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

literalt
z3_propt::lor(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for (unsigned int i = 0; i < bv.size(); i++)
    args[i] = z3_literal(bv[i]);

  result = Z3_mk_or(z3_ctx, bv.size(), args);

  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

literalt
z3_propt::land(literalt a, literalt b)
{
#if 1
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;
#endif
  literalt l = new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_and(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

literalt
z3_propt::lor(literalt a, literalt b)
{
#if 1
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);
  if (a == b) return a;
#endif
  literalt l = new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_or(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

literalt
z3_propt::lnot(literalt a)
{
  a.invert();

  return a;
}

literalt
z3_propt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
z3_propt::new_variable()
{
  literalt l;

  l.set(_no_variables, false);

  set_no_variables(_no_variables + 1);

  return l;
}

void
z3_propt::eliminate_duplicates(const bvt &bv, bvt &dest)
{
  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin(); it != bv.end(); it++)
  {
    if (s.insert(*it).second)
      dest.push_back(*it);
  }
}

bool
z3_propt::process_clause(const bvt &bv, bvt &dest)
{

  dest.clear();

  // empty clause! this is UNSAT
  if (bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin();
       it != bv.end();
       it++)
  {
    literalt l = *it;

    // we never use index 0
    assert(l.var_no() != 0);

    if (l.is_true())
      return true;  // clause satisfied

    if (l.is_false())
      continue;

    assert(l.var_no() < _no_variables);

    // prevent duplicate literals
    if (s.insert(l).second)
      dest.push_back(l);

    if (s.find(lnot(l)) != s.end())
      return true;  // clause satisfied
  }

  return false;
}

void
z3_propt::lcnf(const bvt &bv)
{

  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.size() == 0)
    return;

  Z3_ast lor_var, *args = (Z3_ast*)alloca(new_bv.size() * sizeof(Z3_ast));
  unsigned int i = 0;

  for (bvt::const_iterator it = new_bv.begin(); it != new_bv.end(); it++, i++)
    args[i] = z3_literal(*it);

  if (i > 1) {
    lor_var = Z3_mk_or(z3_ctx, i, args);
    assert_formula(lor_var);
  } else   {
    assert_formula(args[0]);
  }
}

Z3_ast
z3_propt::z3_literal(literalt l)
{

  Z3_ast literal_l;
  std::string literal_s;

  if (l == const_literal(false))
    return Z3_mk_false(z3_ctx);
  else if (l == const_literal(true))
    return Z3_mk_true(z3_ctx);

  literal_s = "l" + i2string(l.var_no());
  literal_l = z3_api.mk_bool_var(literal_s.c_str());

  if (l.sign()) {
    return Z3_mk_not(z3_ctx, literal_l);
  }

  return literal_l;
}

propt::resultt
z3_propt::prop_solve()
{
  return P_ERROR;
}

tvt
z3_propt::l_get(literalt a) const
{
  tvt result = tvt(tvt::TV_ASSUME);
  std::string literal;

  if (a.is_true()) {
    return tvt(true);
  } else if (a.is_false())    {
    return tvt(false);
  }

  symbol_exprt sym("l" + i2string(a.var_no()), bool_typet());
  exprt res = owner.get(sym);

  if (res.is_true())
    result = tvt(true);
  else if (res.is_false())
    result = tvt(false);
  else
    result = tvt(tvt::TV_UNKNOWN);

  if (a.sign())
    result = !result;

  return result;
}

void
z3_propt::assert_formula(Z3_ast ast, bool needs_literal)
{

  // If we're not going to be using the assumptions (ie, for unwidening and for
  // smtlib) then just assert the fact to be true.
  if (!store_assumptions) {
    Z3_assert_cnstr(z3_ctx, ast);
    return;
  }

  if (!needs_literal) {
    Z3_assert_cnstr(z3_ctx, ast);
    assumpt.push_front(ast);
  } else {
    literalt l = new_variable();
    Z3_ast formula = Z3_mk_iff(z3_ctx, z3_literal(l), ast);
    Z3_assert_cnstr(z3_ctx, formula);

    if (smtlib)
      assumpt.push_front(ast);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}

void
z3_propt::assert_literal(literalt l, Z3_ast formula)
{

  Z3_assert_cnstr(z3_ctx, formula);
  if (store_assumptions) {
    if (smtlib)
      assumpt.push_front(formula);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}
