/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <std_code.h>

#include "value_set_domain.h"

/*******************************************************************\

Function: value_set_domaint::transform

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_set_domaint::transform(
  const namespacet &ns,
  locationt from_l,
  locationt to_l)
{
  switch(from_l->type)
  {
  case GOTO:
    // ignore for now
    break;

  case END_FUNCTION:    
    {
    value_set.do_end_function(get_return_lhs(to_l), ns);
    }
    break;
  
  case RETURN:
  case OTHER:
  case ASSIGN:
    {
      expr2tc code = from_l->code;
      value_set.apply_code(code, ns);
    }
    break;

  case FUNCTION_CALL:
    {
      const code_function_call2t &code = to_code_function_call2t(from_l->code);

      const std::vector<expr2tc> &arguments = code.operands;
      value_set.do_function_call(to_l->function, arguments, ns);
    }
    break;
  
  default:;
    // do nothing
  }
}
