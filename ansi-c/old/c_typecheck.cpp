/*******************************************************************\

Module: ANSI-C Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "c_typecheck.h"

/*******************************************************************\

Function: c_typecheckt::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheckt::typecheck()
{
  for(std::list<symbolt>::iterator it=symbols.begin();
      it!=symbols.end();
      it++)
    typecheck_symbol(*it);
}

/*******************************************************************\

Function: c_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_typecheck(
  std::list<symbolt> &symbols,
  contextt &context,
  message_handlert &message_handler,
  const std::string &module)
{
  c_typecheckt c_typecheck(symbols, context, module, message_handler);
  return c_typecheck.typecheck_main();
}

/*******************************************************************\

Function: c_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_typecheck(
  exprt &expr,
  const contextt &context,
  message_handlert &message_handler,
  const std::string &module)
{
  std::list<symbolt> symbols;
  
  contextt tmp_context;

  c_typecheckt c_typecheck(
    symbols, tmp_context, context, module, message_handler);

  try
  {
    c_typecheck.typecheck_expr(expr);
  }

  catch(int e)
  {
    c_typecheck.error();
  }

  catch(const char *e)
  {
    c_typecheck.error(e);
  }

  catch(const std::string &e)
  {
    c_typecheck.error(e);
  }

  return c_typecheck.get_error_found();
}
