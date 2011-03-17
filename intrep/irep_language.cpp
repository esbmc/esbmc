/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include "lispexpr.h"
#include "lispirep.h"
#include "irep_language.h"

/*******************************************************************\

Function: irep_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool irep_languaget::parse(
  std::istream &instream,
  const std::string &path,
  message_handlert &message_handler)
{
  message_handler.print(1, "not implemented");
  return true;
}
             
/*******************************************************************\

Function: irep_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool irep_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  message_handler.print(1, "not implemented");
  return true;
}

/*******************************************************************\

Function: irep_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void irep_languaget::show_parse(std::ostream &out)
{
}

/*******************************************************************\

Function: irep_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool irep_languaget::to_expr(
  const std::string &code,
  const std::string &module,
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  messaget message(message_handler);
  message.error("not yet implemented");
  return true;
}

/*******************************************************************\

Function: new_irep_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_intrep_language()
{
  return new irep_languaget;
}
