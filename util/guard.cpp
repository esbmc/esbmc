/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_expr.h"

#include "guard.h"

/*******************************************************************\

Function: guardt::as_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc guardt::as_expr(guard_listt::const_iterator it) const
{
  if (it == guard_list.end())
    return expr2tc(new constant_bool2t(true));
  else if (it == --guard_list.end())
    return guard_list.back();

  // We can assume at least two operands;
  expr2tc arg1, arg2;
  arg1 = *it++;
  arg2 = *it++;
  expr2tc res = expr2tc(new and2t(arg1, arg2));
  while (it != guard_list.end())
    res = expr2tc(new and2t(res, *it++));

  return res;
}

/*******************************************************************\

Function: guardt::add

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void guardt::add(const expr2tc &expr)
{
  if (is_and2t(expr))
  {
    const and2t &theand = to_and2t(expr);
    add(theand.side_1);
    add(theand.side_2);
    return;
  }

  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).constant_value)
  {
  }
  else
  {
    guard_list.push_back(expr);
  }
}

/*******************************************************************\

Function: guardt::move

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void guardt::move(expr2tc &expr)
{
  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).constant_value)
  {
  }
  else
  {
    guard_list.push_back(expr);
  }
}

/*******************************************************************\

Function: operator -=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: operator |=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
  exprt and_expr1, and_expr2;
  and_expr1 = migrate_expr_back(g1.as_expr(it1));
  and_expr2 = migrate_expr_back(g2.as_expr(it2));
  
  g1.guard_list.erase(it1, g1.guard_list.end());
  
  exprt tmp(and_expr2);
  tmp.make_not();
  
  if(tmp!=and_expr1)
  {
    if(and_expr1.is_true() || and_expr2.is_true())
    {
    }
    else
    {
      exprt or_expr("or", typet("bool"));
      or_expr.move_to_operands(and_expr1, and_expr2);
      expr2tc tmp_expr;
      migrate_expr(or_expr, tmp_expr);
      g1.move(tmp_expr);
    }
  }
  
  return g1;
}

/*******************************************************************\

Function: operator <<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator << (std::ostream &out, const guardt &g)
{
  for (std::list<expr2tc>::const_iterator it = g.guard_list.begin();
       it != g.guard_list.end(); it++)
    out << "*** " << (*it)->pretty() << std::endl;

  return out;
}

/*******************************************************************\

Function: guardt::is_false

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool guardt::is_false() const
{
  forall_guard(it, guard_list)
    if (is_constant_bool2t(*it) && !to_constant_bool2t(*it).constant_value)
      return true;
      
  return false;
}

void
guardt::dump(void) const
{
  std::cout << *this;
  return;
}
