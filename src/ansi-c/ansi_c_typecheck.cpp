/*******************************************************************\

Module: ANSI-C Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_typecheck.h>

/*******************************************************************\

Function: ansi_c_typecheckt::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_typecheckt::typecheck()
{
  for(ansi_c_parse_treet::declarationst::iterator
      it=parse_tree.declarations.begin();
      it!=parse_tree.declarations.end();
      it++)
  {
    symbolt symbol;
    it->to_symbol(symbol);
    typecheck_symbol(symbol);
  }
}

/*******************************************************************\

Function: ansi_c_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_typecheck(
  ansi_c_parse_treet &ansi_c_parse_tree,
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  ansi_c_typecheckt ansi_c_typecheck(
    ansi_c_parse_tree, context, module, message_handler);
  return ansi_c_typecheck.typecheck_main();
}

/*******************************************************************\

Function: ansi_c_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_typecheck(
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  contextt context1, context2;
  ansi_c_parse_treet ansi_c_parse_tree;

  context1 = ns.get_context();

#if 0
  ansi_c_typecheckt ansi_c_typecheck(
    ansi_c_parse_tree, context,
    ns.get_context(), "", message_handler);
#endif
  ansi_c_typecheckt ansi_c_typecheck(
    ansi_c_parse_tree, context1,
    context2, "", message_handler);

  try
  {
    ansi_c_typecheck.typecheck_expr(expr);
  }

  catch(int e)
  {
    ansi_c_typecheck.error();
  }

  catch(const char *e)
  {
    ansi_c_typecheck.error(e);
  }

  catch(const std::string &e)
  {
    ansi_c_typecheck.error(e);
  }
  
  return ansi_c_typecheck.get_error_found();
}
