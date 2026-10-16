#include <goto-symex/engine/goto_symex.h>
#include <irep2/irep2_utils.h>
#include <util/arith/ieee_float.h>
#include <util/arith/transcendental.h>
#include <util/base/host_rounding_mode.h>

#include <cfenv>
#include <cmath>

static expr2tc double_constant(const type2tc &type, double value)
{
  ieee_floatt f(ieee_float_spect(to_floatbv_type(type)));
  f.from_double(value);
  return constant_floatbv2tc(f);
}

static expr2tc between(const expr2tc &value, double low, double high)
{
  return and2tc(
    greaterthanequal2tc(value, double_constant(value->type, low)),
    lessthanequal2tc(value, double_constant(value->type, high)));
}

void goto_symext::symex_bounded_math(
  const code_function_call2t &call,
  const std::string &function)
{
  const expr2tc &ret = call.ret;
  if (is_nil_expr(ret))
    return;
  // The host's values must be those of the default rounding mode.
  const host_rounding_mode rounding_guard(FE_TONEAREST);
  expr2tc x = call.operands[0];
  cur_state->rename(x);
  do_simplify(x);
  if (is_constant_floatbv2t(x))
  {
    const double value = to_constant_floatbv2t(x).value.to_double();
    const double host = transcendental::host_value(function, value);
    const transcendental::interval result =
      is_true(call.operands[1]) ? transcendental::interval{host, host}
                                : transcendental::at(function, value);
    if (result.low == result.high || std::isnan(result.low))
    {
      symex_assign(code_assign2tc(ret, double_constant(ret->type, result.low)));
      return;
    }
    symex_assign(code_assign2tc(ret, gen_nondet(ret->type)));
    assume(between(ret, result.low, result.high));
    return;
  }
  symex_assign(code_assign2tc(ret, gen_nondet(ret->type)));
  expr2tc bounds = gen_true_expr();
  for (const transcendental::piece &p : transcendental::pieces(function))
    bounds = and2tc(
      bounds,
      or2tc(
        not2tc(between(call.operands[0], p.from, p.to)),
        between(ret, p.result.low, p.result.high)));
  assume(bounds);
}
