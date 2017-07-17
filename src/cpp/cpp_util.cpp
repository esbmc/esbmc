/*******************************************************************\

Module:

Author:

\*******************************************************************/

#include <cpp/cpp_util.h>
#include <util/expr.h>
#include <util/symbol.h>

exprt cpp_symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.type);
  tmp.identifier(symbol.name);

  if(symbol.lvalue)
    tmp.set("#lvalue", true);

  return tmp;
}
