#pragma once

#include <irep2/irep2_expr.h>
#include <util/arith/ieee_float.h>

// Double-precision literals for the irep2 identity corpora. A float literal is
// identified by its bit pattern: +0.0 and -0.0 are distinct nodes although IEEE
// equates them, and two NaNs of one sign are one node although IEEE compares
// every NaN unequal.
inline expr2tc float_const(double d, bool negate = false)
{
  ieee_floatt f(ieee_float_spect::double_precision());
  f.from_double(d);
  if (negate)
    f.negate();
  return constant_floatbv2tc(f);
}

inline expr2tc float_nan()
{
  return constant_floatbv2tc(
    ieee_floatt::NaN(ieee_float_spect::double_precision()));
}

inline expr2tc float_inf(bool negative)
{
  const ieee_float_spect spec = ieee_float_spect::double_precision();
  return constant_floatbv2tc(
    negative ? ieee_floatt::minus_infinity(spec)
             : ieee_floatt::plus_infinity(spec));
}
