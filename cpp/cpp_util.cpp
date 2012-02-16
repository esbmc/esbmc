/*******************************************************************\

Module:

Author:

\*******************************************************************/

#include <expr.h>
#include <symbol.h>

#include "cpp_util.h"

/*******************************************************************\

Function: cpp_symbol_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt cpp_symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.type);
  tmp.identifier(symbol.name);

  if(symbol.lvalue)
    tmp.set("#lvalue", true);

  return tmp;
}
