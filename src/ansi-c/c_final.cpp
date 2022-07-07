/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/c_final.h>
#include <c2goto/cprover_library.h>

void c_finalize_expression(const contextt &context, exprt &expr)
{
  std::ostringstream str;
  if(expr.id() == "symbol")
  {
    if(expr.type().id() == "incomplete_array")
    {
      const symbolt *s = context.find_symbol(expr.identifier());

      if(s == nullptr)
      {
        str << "failed to find symbol " << expr.identifier();
        log_error(str.str());
        throw 0;
      }

      const symbolt &symbol = *s;

      if(symbol.type.is_array())
        expr.type() = symbol.type;
      else if(symbol.type.id() == "incomplete_array")
      {
        symbol.location.dump();
        str << "symbol `" << symbol.name << "' has incomplete type";
        log_error(str.str());
        throw 0;
      }
      else
      {
        symbol.location.dump();
        str << "symbol `" << symbol.name << "' has unexpected type";
        log_error(str.str());
        throw 0;
      }
    }
  }

  if(expr.has_operands())
    Forall_operands(it, expr)
      c_finalize_expression(context, *it);
}

bool c_final(contextt &context)
{
  add_cprover_library(context);

  try
  {
    context.Foreach_operand([&context](symbolt &s) {
      if(s.mode == "C")
      {
        c_finalize_expression(context, s.value);
      }
    });
  }

  catch(int e)
  {
    return true;
  }

  return false;
}
