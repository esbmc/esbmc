/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ansi-c/c_final.h>

#include "cpp_final.h"

/*******************************************************************\

Function: cpp_final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_final(
  contextt &context,
  message_handlert &message_handler)
{
  try
  {
    Forall_symbols(it, context.symbols)
    {
      symbolt &symbol=it->second;

      if(symbol.mode=="C++" ||
         symbol.mode=="C")
        c_finalize_expression(context, symbol.value, message_handler);
    }
  }

  catch(int e)
  {
    return true;
  }

  return false;
}
