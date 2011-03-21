/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <sstream>

#include "cvc_language.h"
#include "cvc_typecheck.h"
#include "expr2cvc.h"

/*******************************************************************\

Function: cvc_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cvc_languaget::parse(
  std::istream &instream __attribute__((unused)),
  const std::string &path __attribute__((unused)),
  message_handlert &message_handler)
{
  message_handler.print(1, "not implemented");
  return true;
}
             
/*******************************************************************\

Function: cvc_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cvc_languaget::typecheck(
  contextt &context __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  message_handlert &message_handler)
{
  message_handler.print(1, "not implemented");
  return true;
}

/*******************************************************************\

Function: cvc_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void cvc_languaget::show_parse(std::ostream &out __attribute__((unused)))
{

}

/*******************************************************************\

Function: cvc_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cvc_languaget::from_expr(const exprt &expr, std::string &code,
                              const namespacet &ns __attribute__((unused)))
{
  return expr2cvc(expr, code);
}

/*******************************************************************\

Function: cvc_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cvc_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns __attribute__((unused)))
{
  return type2cvc(type, code);
}

/*******************************************************************\

Function: cvc_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cvc_languaget::to_expr(
  const std::string &code __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  exprt &expr __attribute__((unused)),
  message_handlert &message_handler,
  const namespacet &ns __attribute__((unused)))
{
  messaget message(message_handler);
  message.error("not yet implemented");
  return true;
}

/*******************************************************************\

Function: new_cvc_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
languaget *new_cvc_language()
{
  return new cvc_languaget;
}

