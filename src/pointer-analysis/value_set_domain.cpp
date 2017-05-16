/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <pointer-analysis/value_set_domain.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/std_code.h>

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
    value_set->do_end_function(get_return_lhs(to_l));
  }
  break;

  case RETURN:
  case OTHER:
  case ASSIGN:
    {
      expr2tc code = from_l->code;
      value_set->apply_code(code);
    }
    break;

  case FUNCTION_CALL:
    {
      const code_function_call2t &code = to_code_function_call2t(from_l->code);
      const symbolt &symbol = ns.lookup(to_l->function);

      const std::vector<expr2tc> &arguments = code.operands;
      value_set->do_function_call(symbol, arguments);
    }
    break;

  default:;
    // do nothing
  }
}
