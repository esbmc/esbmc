/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <message_stream.h>

#include "c_final.h"
#include "cprover_library.h"

/*******************************************************************\

Function: c_final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_finalize_expression(
  const contextt &context,
  exprt &expr,
  message_handlert &message_handler)
{
  if(expr.id()=="symbol")
  {
    if(expr.type().id()=="incomplete_array")
    {
      symbolst::const_iterator it=
        context.symbols.find(expr.identifier());

      if(it==context.symbols.end())
      {
        message_streamt message_stream(message_handler);
        message_stream.str
          << "failed to find symbol "
          << expr.identifier();
        message_stream.error();
        throw 0;
      }
      
      const symbolt &symbol=it->second;

      if(symbol.type.is_array())
        expr.type()=symbol.type;
      else if(symbol.type.id()=="incomplete_array")
      {
        message_streamt message_stream(message_handler);
        message_stream.err_location(symbol.location);
        message_stream.str
          << "symbol `" << symbol.display_name()
          << "' has incomplete type";
        message_stream.error();
        throw 0;            
      }
      else
      {
        message_streamt message_stream(message_handler);
        message_stream.err_location(symbol.location);
        message_stream.str
          << "symbol `" << symbol.display_name()
          << "' has unexpected type";
        message_stream.error();
        throw 0;      
      }
    }
  }

  if(expr.has_operands())
    Forall_operands(it, expr)
      c_finalize_expression(context, *it, message_handler);
}

/*******************************************************************\

Function: c_final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_final(contextt &context, message_handlert &message_handler)
{
  add_cprover_library(context, message_handler);

  try
  {
    Forall_symbols(it, context.symbols)
    {
      symbolt &symbol=it->second;

      if(symbol.mode=="C")
      {
        c_finalize_expression(context, symbol.value, message_handler);
      }
    }
  }

  catch(int e)
  {
    return true;
  }

  return false;
}
