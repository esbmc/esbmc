#include <pointer-analysis/value_set_domain.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/std_code.h>

void value_set_domaint::transform(
  const namespacet &ns,
  locationt from_l,
  locationt to_l,
  const std::vector<expr2tc> &arguments)
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
    const symbolt &symbol = *ns.lookup(to_l->function);

    value_set->do_function_call(symbol, arguments);
  }
  break;

  default:;
    // do nothing
  }
}
