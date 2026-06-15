#include "irep2/irep2_expr.h"
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_fp_rounding_utils.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/message/format.h>

// Floating-point specific SMT conversion code extracted from smt_conv.cpp.
// Keep this file focused on IEEE754 constants and FP predicates.
// The rounding-mode enclosure helpers and apply_ieee754_semantics live in
// ir_ieee_conv.cpp (class ir_ieee_convt).

smt_astt smt_convt::get_zero_real()
{
  // Returns SMT representation of zero (0.0)
  return mk_smt_real("0");
}

smt_astt smt_convt::get_double_min_normal()
{
  // IEEE 754 double precision minimum normal positive value (2^-1022)
  return mk_smt_real("2.2250738585072014e-308");
}

smt_astt smt_convt::get_double_min_subnormal()
{
  // IEEE 754 double precision minimum positive subnormal value (2^-1074)
  // Rounded UP at the last digit to ensure the enclosure is conservative.
  return mk_smt_real("4.9406564584124655e-324");
}

smt_astt smt_convt::get_double_max_normal()
{
  // IEEE 754 double: (2^53-1)*2^971 (exact rational, no rounding)
  static const std::string val =
    integer2string((power(2, 53) - 1) * power(2, 971));
  return mk_smt_real(val);
}

smt_astt smt_convt::get_single_min_normal()
{
  // IEEE 754 single precision minimum normal positive value (2^-126)
  return mk_smt_real("1.1754943508222875e-38");
}

smt_astt smt_convt::get_single_min_subnormal()
{
  // IEEE 754 single precision minimum positive subnormal value (2^-149)
  return mk_smt_real("1.4012984643248171e-45");
}

smt_astt smt_convt::get_single_max_normal()
{
  // IEEE 754 single: (2^24-1)*2^104 (exact rational, no rounding)
  static const std::string val =
    integer2string((power(2, 24) - 1) * power(2, 104));
  return mk_smt_real(val);
}

smt_astt smt_convt::get_double_inf_sentinel()
{
  // One above double max_normal: used to represent ±∞ in integer encoding.
  // This value satisfies |x| > max_normal, so isinf/isfinite predicates fire.
  static const std::string val =
    integer2string((power(2, 53) - 1) * power(2, 971) + 1);
  return mk_smt_real(val);
}

smt_astt smt_convt::get_single_inf_sentinel()
{
  // One above single max_normal: used to represent ±∞ in integer encoding.
  static const std::string val =
    integer2string((power(2, 24) - 1) * power(2, 104) + 1);
  return mk_smt_real(val);
}

smt_astt smt_convt::get_double_eps_rel()
{
  // Relative error bound for IEEE 754 double under round-to-nearest:
  // half machine epsilon = 2^-53; rounded UP at the last digit to ensure
  // the enclosure is conservative.
  return mk_smt_real("1.1102230246251566e-16");
}

smt_astt smt_convt::get_single_eps_rel()
{
  // Relative error bound for IEEE 754 single under round-to-nearest:
  // half machine epsilon = 2^-24
  return mk_smt_real("5.960464477539063e-08");
}

smt_astt smt_convt::get_double_eps_up()
{
  // B_dir relative error bound for IEEE 754 double under round-toward-+inf:
  // eps_rel_dir = 2^-52 (the full machine epsilon for double, DBL_EPSILON).
  // Conservative decimal: strictly greater than 2^-52 to preserve soundness.
  return mk_smt_real("2.2204460492503131e-16");
}

smt_astt smt_convt::get_single_eps_up()
{
  // B_dir relative error bound for IEEE 754 single under round-toward-+inf:
  // eps_rel_dir = 2^-23 (the full machine epsilon for single, FLT_EPSILON).
  // Exact decimal representation of 2^-23; no rounding required.
  return mk_smt_real("1.1920928955078125e-07");
}

smt_astt smt_convt::convert_is_nan(const expr2tc &expr)
{
  const isnan2t &isnan = to_isnan2t(expr);

  // Anything other than floats will never be NaNs
  if (!is_floatbv_type(isnan.value) || int_encoding)
    return mk_smt_bool(false);

  smt_astt operand = convert_ast(isnan.value);
  return fp_api->mk_smt_fpbv_is_nan(operand);
}

// Returns the max-normal SMT real for single/double floatbv, or nullptr for
// any other format (caller must handle the unsupported case explicitly).
static smt_astt get_max_normal(smt_convt &conv, const floatbv_type2t &fbv_type)
{
  const auto single_spec = ieee_float_spect::single_precision();
  const auto double_spec = ieee_float_spect::double_precision();
  if (fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
    return conv.get_single_max_normal();
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
    return conv.get_double_max_normal();
  return nullptr;
}

// Returns the min-normal SMT real for single/double floatbv, or nullptr.
static smt_astt get_min_normal(smt_convt &conv, const floatbv_type2t &fbv_type)
{
  const auto single_spec = ieee_float_spect::single_precision();
  const auto double_spec = ieee_float_spect::double_precision();
  if (fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
    return conv.get_single_min_normal();
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
    return conv.get_double_min_normal();
  return nullptr;
}

smt_astt smt_convt::convert_is_inf(const expr2tc &expr)
{
  const isinf2t &isinf = to_isinf2t(expr);

  // Anything other than floats will never be infs
  if (!is_floatbv_type(isinf.value))
    return mk_smt_bool(false);

  if (int_encoding)
  {
    // In integer/real encoding a float is "infinite" when its real value
    // exceeds the maximum finite float magnitude: |f| > max_normal.
    const floatbv_type2t &fbv_type = to_floatbv_type(isinf.value->type);
    smt_astt max_val = get_max_normal(*this, fbv_type);
    if (!max_val)
    {
      log_warning(
        "isinf: unsupported float format (exp={}, frac={}); returning false",
        fbv_type.exponent,
        fbv_type.fraction);
      return mk_smt_bool(false);
    }
    smt_astt operand = convert_ast(isinf.value);
    smt_astt pos_inf = mk_gt(operand, max_val);
    smt_astt neg_inf = mk_lt(operand, mk_sub(get_zero_real(), max_val));
    return mk_or(pos_inf, neg_inf);
  }

  smt_astt operand = convert_ast(isinf.value);
  return fp_api->mk_smt_fpbv_is_inf(operand);
}

smt_astt smt_convt::convert_is_normal(const expr2tc &expr)
{
  const isnormal2t &isnormal = to_isnormal2t(expr);

  // Anything other than floats will always be normal
  if (!is_floatbv_type(isnormal.value))
    return mk_smt_bool(true);

  if (int_encoding)
  {
    // In integer/real encoding a float is normal when min_normal <= |f| <=
    // max_normal.  Zero and subnormals (|f| < min_normal) are not normal.
    const floatbv_type2t &fbv_type = to_floatbv_type(isnormal.value->type);
    smt_astt max_val = get_max_normal(*this, fbv_type);
    smt_astt min_val = get_min_normal(*this, fbv_type);
    if (!max_val || !min_val)
    {
      log_warning(
        "isnormal: unsupported float format (exp={}, frac={}); returning true",
        fbv_type.exponent,
        fbv_type.fraction);
      return mk_smt_bool(true);
    }
    smt_astt zero = get_zero_real();
    smt_astt neg_max = mk_sub(zero, max_val);
    smt_astt neg_min = mk_sub(zero, min_val);
    smt_astt operand = convert_ast(isnormal.value);
    // |f| >= min_normal: f >= min_normal || f <= -min_normal
    smt_astt above_min =
      mk_or(mk_ge(operand, min_val), mk_le(operand, neg_min));
    // |f| <= max_normal: f <= max_normal && f >= -max_normal
    smt_astt below_max =
      mk_and(mk_le(operand, max_val), mk_ge(operand, neg_max));
    return mk_and(above_min, below_max);
  }

  smt_astt operand = convert_ast(isnormal.value);
  return fp_api->mk_smt_fpbv_is_normal(operand);
}

smt_astt smt_convt::convert_is_finite(const expr2tc &expr)
{
  const isfinite2t &isfinite = to_isfinite2t(expr);

  // Anything other than floats will always be finite
  if (!is_floatbv_type(isfinite.value))
    return mk_smt_bool(true);

  if (int_encoding)
  {
    // In integer/real encoding a float is finite when |f| <= max_normal.
    const floatbv_type2t &fbv_type = to_floatbv_type(isfinite.value->type);
    smt_astt max_val = get_max_normal(*this, fbv_type);
    if (!max_val)
    {
      log_warning(
        "isfinite: unsupported float format (exp={}, frac={}); returning true",
        fbv_type.exponent,
        fbv_type.fraction);
      return mk_smt_bool(true);
    }
    smt_astt operand = convert_ast(isfinite.value);
    smt_astt pos_ok = mk_le(operand, max_val);
    smt_astt neg_ok = mk_ge(operand, mk_sub(get_zero_real(), max_val));
    return mk_and(pos_ok, neg_ok);
  }

  smt_astt value = convert_ast(isfinite.value);

  // isfinite = !(isinf || isnan)
  smt_astt isinf = fp_api->mk_smt_fpbv_is_inf(value);
  smt_astt isnan = fp_api->mk_smt_fpbv_is_nan(value);

  smt_astt or_op = mk_or(isinf, isnan);
  return mk_not(or_op);
}

smt_astt smt_convt::convert_signbit(const expr2tc &expr)
{
  const signbit2t &signbit = to_signbit2t(expr);

  // Extract the top bit
  auto value = convert_ast(signbit.operand);

  smt_astt is_neg;

  if (int_encoding)
  {
    // In integer/real encoding mode, floating-point values are represented as reals
    // We can't extract bits, so check the sign mathematically
    is_neg = mk_lt(value, mk_smt_real("0"));
  }
  else
  {
    // In bitvector mode, extract the sign bit
    const auto width = value->sort->get_data_width();
    is_neg =
      mk_eq(mk_extract(value, width - 1, width - 1), mk_smt_bv(BigInt(1), 1));
  }

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(
    is_neg,
    convert_ast(gen_one(signbit.type)),
    convert_ast(gen_zero(signbit.type)));
}

smt_astt smt_convt::convert_rounding_mode(const expr2tc &expr)
{
  // We don't actually care about rounding mode when in integer/real mode, as
  // it is discarded when encoding it in SMT
  if (int_encoding)
    return nullptr;

  // Easy case, we know the rounding mode
  if (is_constant_int2t(expr))
  {
    const int64_t raw_rm = to_constant_int2t(expr).value.to_int64();
    switch (raw_rm)
    {
    case ieee_floatt::ROUND_TO_EVEN:
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_EVEN);
    case ieee_floatt::ROUND_TO_AWAY:
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_AWAY);
    case ieee_floatt::ROUND_TO_PLUS_INF:
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_PLUS_INF);
    case ieee_floatt::ROUND_TO_MINUS_INF:
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_MINUS_INF);
    case ieee_floatt::ROUND_TO_ZERO:
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
    default:
      // Preserve the historical fallback behavior of this conversion path.
      log_warning(
        "unsupported constant rounding mode value {}: falling back to "
        "ROUND_TO_ZERO",
        raw_rm);
      return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
    }
  }

  assert(is_symbol2t(expr));
  // Symbolic values follow ieee_floatt::rounding_modet encoding:
  // ROUND_TO_EVEN=0, ROUND_TO_AWAY=1, ROUND_TO_PLUS_INF=2,
  // ROUND_TO_MINUS_INF=3, ROUND_TO_ZERO=4.

  smt_astt symbol = convert_ast(expr);
  if (symbol->sort->id != SMT_SORT_BV)
  {
    log_warning(
      "unsupported symbolic rounding mode sort {}: falling back to "
      "ROUND_TO_ZERO",
      static_cast<int>(symbol->sort->id));
    return fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  }

  const auto width = symbol->sort->get_data_width();

  auto is_mode = [this, &symbol, width](int value) -> smt_astt {
    return mk_eq(symbol, mk_smt_bv(BigInt(value), width));
  };

  smt_astt is_0 = is_mode(ieee_floatt::ROUND_TO_EVEN);
  smt_astt is_1 = is_mode(ieee_floatt::ROUND_TO_AWAY);
  smt_astt is_2 = is_mode(ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt is_3 = is_mode(ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt is_4 = is_mode(ieee_floatt::ROUND_TO_ZERO);

  smt_astt ne = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_EVEN);
  smt_astt na = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_AWAY);
  smt_astt mi = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt pi = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt ze = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);

  // Keep an explicit map for all supported modes:
  // 0 -> nearest-even, 1 -> nearest-away, 2 -> +inf, 3 -> -inf, 4 -> zero.
  // Any other symbolic value conservatively falls back to ROUND_TO_ZERO, which
  // matches the previous behavior of this conversion path.
  smt_astt ite4 = ze;
  ite4 = mk_ite(is_4, ze, ite4);
  smt_astt ite3 = mk_ite(is_3, mi, ite4);
  smt_astt ite2 = mk_ite(is_2, pi, ite3);
  smt_astt ite1 = mk_ite(is_1, na, ite2);
  smt_astt ite0 = mk_ite(is_0, ne, ite1);

  return ite0;
}
