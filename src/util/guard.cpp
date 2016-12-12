/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_expr.h"

#include "guard.h"

expr2tc guardt::as_expr(guard_listt::const_iterator it) const
{
  if (it == guard_list.end())
    return true_expr;
  else if (it == --guard_list.end())
    return guard_list.back();

  // We can assume at least two operands;
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
  if (is_and2t(expr))
  {
    const and2t &theand = to_and2t(expr);
    add(theand.side_1);
    add(theand.side_2);
    return;
  }

  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).value)
  {
  }
  else
  {
    guard_list.push_back(expr);
  }
}

void guardt::move(expr2tc &expr)
{
  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).value)
  {
  }
  else
  {
    guard_list.push_back(expr);
  }
}

guardt &operator -= (guardt &g1, const guardt &g2)
{
  guardt::guard_listt::const_iterator it2=g2.guard_list.begin();
  
  while(!g1.guard_list.empty() &&
        it2!=g2.guard_list.end() &&
        g1.guard_list.front()==*it2)
  {
    g1.guard_list.pop_front();
    it2++;
  }


  return g1;
}

void
guardt::back_sub(const guardt &g2)
{
  guardt::guard_listt::const_reverse_iterator it2 = g2.guard_list.rbegin();

  while (!guard_list.empty() &&
         it2 != g2.guard_list.rend() &&
         guard_list.back()==*it2)
  {
    guard_list.pop_back();
    it2++;
  }
}

guardt &operator |= (guardt &g1, const guardt &g2)
{
  if(g2.is_false()) return g1;
  if(g1.is_false()) { g1.guard_list=g2.guard_list; return g1; }

  // find common prefix  
  guardt::guard_listt::iterator it1=g1.guard_list.begin();
  guardt::guard_listt::const_iterator it2=g2.guard_list.begin();
  
  while(it1!=g1.guard_list.end())
  {
    if(it2==g2.guard_list.end())
      break;
      
    if(*it1!=*it2)
      break;

    it1++;
    it2++;
  }
  
  if(it2==g2.guard_list.end()) return g1;

  // end of common prefix
  expr2tc and_expr1, and_expr2;
  and_expr1 = g1.as_expr(it1);
  and_expr2 = g2.as_expr(it2);
  
  g1.guard_list.erase(it1, g1.guard_list.end());
  
  not2tc tmp(and_expr2);
  
  if (tmp != and_expr1)
  {
    if ((is_constant_bool2t(and_expr1) &&
         to_constant_bool2t(and_expr1).value) ||
        (is_constant_bool2t(and_expr2) &&
         to_constant_bool2t(and_expr2).value))
    {
    }
    else
    {
      or2tc or_expr(and_expr1, and_expr2);
      g1.move(or_expr);
    }
  }
  
  return g1;
}

std::ostream &operator << (std::ostream &out, const guardt &g)
{
  for (std::list<expr2tc>::const_iterator it = g.guard_list.begin();
       it != g.guard_list.end(); it++)
    out << "*** " << (*it)->pretty() << std::endl;

  return out;
}

bool guardt::is_false() const
{
  forall_guard(it, guard_list)
    if (is_constant_bool2t(*it) && !to_constant_bool2t(*it).value)
      return true;
      
  return false;
}

void
guardt::dump(void) const
{
  std::cout << *this;
  return;
}

bool
operator == (const guardt &g1, const guardt &g2)
{
  // Very simple: the guard list should be identical.
  return g1.guard_list == g2.guard_list;
}
