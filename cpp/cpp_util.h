/*******************************************************************\

Module:

Author:

\*******************************************************************/

#ifndef CPROVER_CPP_UTIL_H
#define CPROVER_CPP_UTIL_H

#include <expr.h>
#include <symbol.h>

/*******************************************************************\

Function: cpp_symbol_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

extern inline exprt cpp_symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.type);
  tmp.set("identifier", symbol.name);
  if(symbol.lvalue)
    tmp.set("#lvalue",true);
  return tmp;
}


/*******************************************************************\

Function: already_typechecked

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/


extern inline void already_typechecked(exprt& expr)
{
    exprt tmp("already_typechecked");
    tmp.copy_to_operands(expr);
    expr.swap(tmp);
}

#endif
