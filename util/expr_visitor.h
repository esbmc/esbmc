/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_UTIL_EXPR_VISITOR
#define CPROVER_UTIL_EXPR_VISITOR

#include "expr.h"

class expr_visitort
{
public:

  void operator() (exprt &expr)
  {
    visitor(expr);
  }

  void operator() (const exprt &expr)
  {
    visitor(expr);
  }
  
  virtual ~expr_visitort()
  {
  }

protected:
  void visitor(exprt &expr)
  {
    pre_visitor(expr);

    Forall_operands(it, expr)
      visitor(*it);
      
    post_visitor(expr);
  }

  virtual void pre_visitor(exprt &expr __attribute__((unused)))
  {
  }

  virtual void post_visitor(exprt &expr __attribute__((unused)))
  {
  }
  
  void visitor(const exprt &expr)
  {
    pre_visitor(expr);

    forall_operands(it, expr)
      visitor(*it);
      
    post_visitor(expr);
  }

  virtual void pre_visitor(const exprt &expr __attribute__((unused)))
  {
  }

  virtual void post_visitor(const exprt &expr __attribute__((unused)))
  {
  }
};

#endif
