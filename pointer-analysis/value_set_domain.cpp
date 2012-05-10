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
    expr2tc ret;
    migrate_expr(get_return_lhs(to_l), ret);
    value_set.do_end_function(ret, ns);
    }
    break;
  
  case RETURN:
  case OTHER:
  case ASSIGN:
    {
      expr2tc code;
      migrate_expr(from_l->code, code);
      value_set.apply_code(code, ns);
    }
    break;

  case FUNCTION_CALL:
    {
      const code_function_callt &code=
        to_code_function_call(from_l->code);

      const exprt::operandst &arguments = code.arguments();
      std::vector<expr2tc> args;
      for (exprt::operandst::const_iterator it = arguments.begin();
           it != arguments.end(); it++) {
        expr2tc tmp_arg;
        migrate_expr(*it, tmp_arg);
        args.push_back(tmp_arg);
      }

      value_set.do_function_call(to_l->function, args, ns);
    }
    break;
  
  default:;
    // do nothing
  }
}
