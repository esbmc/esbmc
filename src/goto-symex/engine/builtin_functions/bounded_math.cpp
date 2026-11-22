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

// __builtin_bounded_<f>(a, b) of hypot or arctan2: arguments constant by now
// get the C library's value within the ulps bound, exact where C99 fixes
// the result; the caller's relations bound any other arguments.
void goto_symext::symex_bounded_binary_math(
  const code_function_call2t &call,
  const std::string &function)
{
  const expr2tc &ret = call.ret;
  expr2tc a = call.operands[0], b = call.operands[1];
  cur_state->rename(a);
  cur_state->rename(b);
  do_simplify(a);
  do_simplify(b);
  symex_assign(code_assign2tc(ret, gen_nondet(ret->type)));
  if (!is_constant_floatbv2t(a) || !is_constant_floatbv2t(b))
    return;
  const double first = to_constant_floatbv2t(a).value.to_double();
  const double second = to_constant_floatbv2t(b).value.to_double();
  const double value = transcendental::host_value(function, first, second);
  const bool special = first == 0.0 || second == 0.0 || !std::isfinite(first) ||
                       !std::isfinite(second);
  const transcendental::interval result =
    special ? transcendental::interval{value, value}
            : transcendental::within_ulps(value);
  if (std::isnan(result.low))
    return;
  assume(between(ret, result.low, result.high));
}

void goto_symext::symex_bounded_math(
  const code_function_call2t &call,
  const std::string &function)
{
  const expr2tc &ret = call.ret;
  if (is_nil_expr(ret))
    return;
  // The host's values must be those of the default rounding mode.
  if (call.operands.size() == 2 && transcendental::is_bounded_binary(function))
  {
    const host_rounding_mode rounding_guard(FE_TONEAREST);
    symex_bounded_binary_math(call, function);
    return;
  }
  // The host's values must be those of the default rounding mode.
  const host_rounding_mode rounding_guard(FE_TONEAREST);
  expr2tc x = call.operands[0];
  cur_state->rename(x);
  do_simplify(x);
  if (is_constant_floatbv2t(x))
  {
    const double value = to_constant_floatbv2t(x).value.to_double();
    const double host = transcendental::host_value(function, value);
    // math.* gives the C library's value, except for the functions CPython
    // computes itself.
    const bool library = is_true(call.operands[1]) &&
                         !transcendental::cpython_implements(function);
    const transcendental::interval result =
      library ? transcendental::interval{host, host}
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
