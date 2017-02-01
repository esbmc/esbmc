/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_expr.h"

#include "guard.h"

expr2tc guardt::as_expr() const
{
  if(is_true())
    return true_expr;

  if(is_single_symbol())
    return *guard_list.begin();

  // We can assume at least two operands, return a chain of ands
  auto it = guard_list.begin();

  expr2tc arg1, arg2;
  arg1 = *it++;
  arg2 = *it++;
  and2tc res(arg1, arg2);
  while (it != guard_list.end())
    res = and2tc(res, *it++);

  return res;
}

void guardt::add(const expr2tc &expr)
{
  if(is_false() || ::is_true(expr))
    return;

  if(is_true() || ::is_false(expr))
  {
    guard_list.clear();
    guard_list.insert(expr);
    return;
  }

  if(is_and2t(expr))
  {
    const and2t &theand = to_and2t(expr);
    add(theand.side_1);
    add(theand.side_2);
    return;
  }

  guard_list.insert(expr);
}

void guardt::append(const guardt &guard)
{
  for(auto it : guard.guard_list)
    add(it);
}

guardt &operator -= (guardt &g1, const guardt &g2)
{
  for(auto it2 : g2.guard_list)
    g1.guard_list.erase(it2);
  return g1;
}

guardt &operator |= (guardt &g1, const guardt &g2)
{
  // Easy cases
  if(g2.is_false() || g1.is_true()) return g1;
  if(g1.is_false() || g2.is_true()) { g1 = g2; return g1; }

  if(g1.is_single_symbol() && g2.is_single_symbol())
  {
    // Both guards have one symbol, so check if we opposite symbols, e.g,
    // g1 == sym1 and g2 == !sym1
    expr2tc or_expr(new or2t(*g1.guard_list.begin(), *g2.guard_list.begin()));
    do_simplify(or_expr);

    if(::is_true(or_expr)) { g1.make_true(); return g1; }

    // Despite if we could simplify or not, clear and set the new guard
    g1.clear_insert(or_expr);
  }
  else
  {
    // Here, we have a symbol (or symbols) in g2 to be or'd with the symbol
    // (or symbols) in g1, e.g:
    // g1 = !guard3 && !guard2 && !guard1
    // g2 = guard2 && !guard1
    // res = g1 || g2 = (!guard3 && !guard2 && !guard1) || (guard2 && !guard1)

    // Simplify equation: everything that's common in both guards, will not
    // be or'd

    // Create a new g2, which we can remove stuff from
    guardt new_g2(g2);

    // Common guards
    guardt common;

    for(auto it2 : g2.guard_list)
    {
      auto it1 = g1.guard_list.find(it2);
      if(it1 != g1.guard_list.end())
      {
        common.add(it2);
        new_g2.guard_list.erase(it2);
        g1.guard_list.erase(it1);
      }
    }

    // Get the and expression from both guards
    expr2tc or_expr(new or2t(g1.as_expr(), new_g2.as_expr()));

    // If the guards single symbols, try to simplify the or expression
    if(g1.is_single_symbol() && new_g2.is_single_symbol())
      do_simplify(or_expr);

    g1.clear_append(common);
    g1.add(or_expr);
  }

  return g1;
}

void
guardt::dump(void) const
{
  for (auto it : guard_list)
    it->dump();
}

bool
operator == (const guardt &g1, const guardt &g2)
{
  // Very simple: the guard list should be identical.
  return g1.guard_list == g2.guard_list;
}

void guardt::swap(guardt& g)
{
  guard_list.swap(g.guard_list);
}

void guardt::clear_insert(const expr2tc& expr)
{
  guard_list.clear();
  add(expr);
}

void guardt::guard_expr(expr2tc& dest) const
{
  // Fills the expr only if it's not true
  if(is_true())
    return;

  if(::is_false(dest))
  {
    dest = as_expr();
    make_not(dest);
    return;
  }

  dest = expr2tc(new implies2t(as_expr(), dest));
}

bool guardt::empty() const
{
  return guard_list.empty();
}

bool guardt::is_true() const
{
  return empty();
}

bool guardt::is_false() const
{
  // Never false
  if(guard_list.size() != 1)
    return false;

  return (*guard_list.begin() == false_expr);
}

void guardt::make_true()
{
  guard_list.clear();
}

void guardt::make_false()
{
  add(false_expr);
}

bool guardt::is_single_symbol() const
{
  return (guard_list.size() == 1);
}

void guardt::clear_append(const guardt& guard)
{
  guard_list.clear();
  append(guard);
}
