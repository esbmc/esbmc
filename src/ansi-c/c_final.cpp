/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <message_stream.h>

#include "c_final.h"
#include <c2goto/cprover_library.h>

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
      const symbolt* s = context.find_symbol(expr.identifier());

      if(s == nullptr)
      {
        message_streamt message_stream(message_handler);
        message_stream.str
          << "failed to find symbol "
          << expr.identifier();
        message_stream.error();
        throw 0;
      }

      const symbolt &symbol = *s;

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
    context.Foreach_operand(
      [&context, &message_handler] (symbolt& s)
      {
        if(s.mode=="C")
        {
          c_finalize_expression(context, s.value, message_handler);
        }
      }
    );
  }

  catch(int e)
  {
    return true;
  }

  return false;
}
