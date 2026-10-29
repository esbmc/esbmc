#include <pointer-analysis/value_set_domain.h>
#include <irep2/irep2.h>
#include <util/irep/migrate.h>
#include <util/irep/std_code.h>

void value_set_domaint::transform(
  const namespacet &ns,
  locationt from_l,
  locationt to_l)
{
  switch (from_l->type)
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
    const symbolt &symbol = *ns.lookup(to_l->function);

    const std::vector<expr2tc> &arguments = code.operands;
    value_set->do_function_call(symbol, arguments);
  }
  break;

  case ASSUME:
    // Refine value-sets by the ASSUME guard. Currently handles `p != 0` to
    // strip null branches from pointer-typed value-sets — see
    // value_sett::apply_assume. Without this VSA treats every ASSUME as
    // no-op, so `__ESBMC_assume(alloc != 0)` after a potentially-failing
    // allocator leaves the null branch in the points-to set for every
    // subsequent read of that pointer.
    value_set->apply_assume(from_l->guard);
    break;

  default:;
    // do nothing
  }
}
