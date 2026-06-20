#include "irep2/irep2_expr.h"
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/smt_fp_rounding_utils.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/message/format.h>

// Floating-point specific SMT conversion code extracted from smt_conv.cpp.
// Keep this file focused on IEEE754 constants/semantics and FP predicates.

smt_astt smt_solver_baset::get_zero_real()
{
  // Returns SMT representation of zero (0.0)
  return mk_smt_real("0");
}

smt_astt smt_solver_baset::get_double_min_normal()
{
  // IEEE 754 double precision minimum normal positive value (2^-1022)
  return mk_smt_real("2.2250738585072014e-308");
}

smt_astt smt_solver_baset::get_double_min_subnormal()
{
  // IEEE 754 double precision minimum positive subnormal value (2^-1074)
  // Rounded UP at the last digit to ensure the enclosure is conservative.
  return mk_smt_real("4.9406564584124655e-324");
}

smt_astt smt_solver_baset::apply_ieee754_semantics(
  smt_astt real_result,
  const floatbv_type2t &fbv_type,
  smt_astt operand_zero_check,
  const expr2tc &rounding_mode)
{
  if (this->options.get_bool_option("ir-ieee"))
  {
    auto weak_enclosure_return =
      [this, &real_result, &fbv_type](const char *reason) -> smt_astt {
      log_warning(
        "using weak IEEE754 enclosure ({}), exp={}, frac={}",
        reason,
        fbv_type.exponent,
        fbv_type.fraction);
      smt_sortt rs = mk_real_sort();
      smt_astt ra_lo = mk_fresh(rs, "ra_lo_weak::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi_weak::", nullptr);
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));
      return real_result;
    };

    auto select_nearest_eps =
      [this, &fbv_type](smt_astt &eps_rel, smt_astt &eps_abs) -> bool {
      const auto double_spec = ieee_float_spect::double_precision();
      const auto single_spec = ieee_float_spect::single_precision();
      if (
        fbv_type.exponent == double_spec.e &&
        fbv_type.fraction == double_spec.f)
      {
        eps_rel = get_double_eps_rel();       // 2^-53
        eps_abs = get_double_min_subnormal(); // 2^-1074
        return true;
      }
      if (
        fbv_type.exponent == single_spec.e &&
        fbv_type.fraction == single_spec.f)
      {
        eps_rel = get_single_eps_rel();       // 2^-24
        eps_abs = get_single_min_subnormal(); // 2^-149
        return true;
      }
      return false;
    };

    auto abs_real = [this](smt_astt r) -> smt_astt {
      smt_astt zero = mk_smt_real("0.0");
      return mk_ite(mk_lt(r, zero), mk_sub(zero, r), r);
    };

    auto select_directed_eps =
      [this, &fbv_type](smt_astt &eps_rel_dir, smt_astt &eps_abs) -> bool {
      const auto double_spec = ieee_float_spect::double_precision();
      const auto single_spec = ieee_float_spect::single_precision();
      if (
        fbv_type.exponent == double_spec.e &&
        fbv_type.fraction == double_spec.f)
      {
        eps_rel_dir = get_double_eps_up();    // 2^-52
        eps_abs = get_double_min_subnormal(); // 2^-1074
        return true;
      }
      if (
        fbv_type.exponent == single_spec.e &&
        fbv_type.fraction == single_spec.f)
      {
        eps_rel_dir = get_single_eps_up();    // 2^-23
        eps_abs = get_single_min_subnormal(); // 2^-149
        return true;
      }
      return false;
    };

    if (smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode))
    {
      // Tight path: rounding mode is concrete round-to-nearest.
      // Attach a sound symmetric enclosure around the real semantic
      // term `r` for nearest rounding mode.
      //
      // For an FP operation whose exact real value is r, the round-to-nearest
      // error satisfies:
      //   |fl(r) - r| <= eps_rel * |r| + eps_abs
      //
      // So we define:
      //   B(r)  = eps_rel * |r| + eps_abs
      //   ra_lo = r - B(r)
      //   ra_hi = r + B(r)
      //
      // and assert  ra_lo <= result <= ra_hi  and  ra_lo <= ra_hi.
      //
      // TODO: extend to non-standard FP formats beyond single and double.

      smt_astt eps_rel = nullptr, eps_abs = nullptr;
      if (!select_nearest_eps(eps_rel, eps_abs))
      {
        // TODO: theorem-driven bounds for non-standard formats are not yet
        // implemented; fall back to unconstrained enclosure (sound but weak).
        return weak_enclosure_return("nearest: unsupported fp format");
      }

      smt_sortt rs = mk_real_sort();
      smt_astt abs_r = abs_real(real_result);

      // B(r) = eps_rel * |r| + eps_abs
      smt_astt bound = mk_add(mk_mul(eps_rel, abs_r), eps_abs);

      // ra_lo = r - B(r),  ra_hi = r + B(r)
      smt_astt ra_lo_expr = mk_sub(real_result, bound);
      smt_astt ra_hi_expr = mk_add(real_result, bound);

      // Introduce named enclosure variables and pin them to the bound expressions
      // via bidirectional inequalities.  A plain mk_eq(ra_lo, ra_lo_expr) is
      // correct but the Z3 tactic pipeline (solve-eqs) eliminates definitional
      // equalities from the visible assertion set, making them invisible in the
      // SMT output.  Two mk_le assertions are logically equivalent and are not
      // targeted by solve-eqs, so they survive in the final formula.
      smt_astt ra_lo = mk_fresh(rs, "ra_lo::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi::", nullptr);
      assert_ast(mk_le(ra_lo, ra_lo_expr)); // ra_lo <= r - B(r)
      assert_ast(
        mk_le(ra_lo_expr, ra_lo)); // r - B(r) <= ra_lo  =>  ra_lo == r - B(r)
      assert_ast(mk_le(ra_hi, ra_hi_expr)); // ra_hi <= r + B(r)
      assert_ast(
        mk_le(ra_hi_expr, ra_hi)); // r + B(r) <= ra_hi  =>  ra_hi == r + B(r)

      // Containment: ra_lo <= result <= ra_hi
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode))
    {
      // Asymmetric tight enclosure for ROUND_TO_PLUS_INF:
      //   fl_RUP(r) >= r  (exact lower bound; round-up never undershoots)
      //   fl_RUP(r) <= r + B_dir(r)
      // where B_dir(r) = eps_rel_dir * |r| + eps_abs
      //   eps_rel_dir = 2^-52 (double) or 2^-23 (single) -- the full machine epsilon

      smt_sortt rs = mk_real_sort();
      smt_astt eps_up, eps_abs;
      if (!select_directed_eps(eps_up, eps_abs))
      {
        // Unsupported format: fall back to unconstrained weak enclosure.
        return weak_enclosure_return(
          "round-to-plus-inf: unsupported fp format");
      }

      smt_astt abs_r = abs_real(real_result);

      // B_dir(r) = eps_rel_dir * |r| + eps_abs
      smt_astt b_dir = mk_add(mk_mul(eps_up, abs_r), eps_abs);
      smt_astt ra_hi_expr = mk_add(real_result, b_dir);

      // Introduce named enclosure variables. Use bidirectional inequalities to
      // survive Z3's solve-eqs tactic (same technique as the nearest-mode path).
      smt_astt ra_lo = mk_fresh(rs, "ra_lo_up::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi_up::", nullptr);

      // Pin ra_lo = r (the exact lower bound for round-up)
      assert_ast(mk_le(ra_lo, real_result)); // ra_lo <= r
      assert_ast(mk_le(real_result, ra_lo)); // r <= ra_lo  =>  ra_lo == r

      // Pin ra_hi = r + B_dir(r)
      assert_ast(mk_le(ra_hi, ra_hi_expr)); // ra_hi <= r + B_dir(r)
      assert_ast(mk_le(ra_hi_expr, ra_hi)); // r + B_dir(r) <= ra_hi

      // Containment: ra_lo <= result <= ra_hi
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode))
    {
      // Asymmetric tight enclosure for ROUND_TO_MINUS_INF:
      //   fl_RDN(r) <= r  (exact upper bound; round-down never overshoots)
      //   fl_RDN(r) >= r - B_dir(r)
      // where B_dir(r) = eps_rel_dir * |r| + eps_abs
      //   eps_rel_dir = 2^-52 (double) or 2^-23 (single)
      //   This is the directed-mode error constant, the same value used for
      //   ROUND_TO_PLUS_INF; the bound shape is the mirror image.

      smt_sortt rs = mk_real_sort();
      smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
      if (!select_directed_eps(eps_rel_dir, eps_abs))
      {
        // Unsupported format: fall back to unconstrained weak enclosure.
        return weak_enclosure_return(
          "round-to-minus-inf: unsupported fp format");
      }

      smt_astt abs_r = abs_real(real_result);

      // B_dir(r) = eps_rel_dir * |r| + eps_abs
      smt_astt b_dir = mk_add(mk_mul(eps_rel_dir, abs_r), eps_abs);
      smt_astt ra_lo_expr = mk_sub(real_result, b_dir); // r - B_dir(r)

      // Introduce named enclosure variables. Use bidirectional inequalities to
      // survive Z3's solve-eqs tactic (same technique as the other tight paths).
      smt_astt ra_lo = mk_fresh(rs, "ra_lo_dn::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi_dn::", nullptr);

      // Pin ra_lo = r - B_dir(r)  (the computed lower bound)
      assert_ast(mk_le(ra_lo, ra_lo_expr)); // ra_lo <= r - B_dir(r)
      assert_ast(mk_le(ra_lo_expr, ra_lo)); // r - B_dir(r) <= ra_lo

      // Pin ra_hi = r  (exact upper bound for round-down)
      assert_ast(mk_le(ra_hi, real_result)); // ra_hi <= r
      assert_ast(mk_le(real_result, ra_hi)); // r <= ra_hi  =>  ra_hi == r

      // Containment: ra_lo <= result <= ra_hi
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_zero(rounding_mode))
    {
      // Asymmetric tight enclosure for ROUND_TO_ZERO (truncation toward zero).
      //
      // RTZ is sign-dependent: it rounds down for r >= 0 and rounds up for r < 0.
      //   r >= 0:  fl_RTZ(r) in [r - B_dir(r),  r]   (same shape as RDN)
      //   r <  0:  fl_RTZ(r) in [r,  r + B_dir(r)]   (same shape as RUP)
      //
      // Unified via ITE on sign:
      //   ra_lo = ite(r >= 0,  r - B_dir(r),  r)
      //   ra_hi = ite(r >= 0,  r,              r + B_dir(r))
      //
      // where B_dir(r) = eps_rel_dir * |r| + eps_abs
      //   eps_rel_dir = 2^-52 (double) or 2^-23 (single) -- full machine epsilon

      smt_sortt rs = mk_real_sort();
      smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
      if (!select_directed_eps(eps_rel_dir, eps_abs))
      {
        // Unsupported format: fall back to unconstrained weak enclosure.
        return weak_enclosure_return("round-to-zero: unsupported fp format");
      }

      smt_astt abs_r = abs_real(real_result);

      smt_astt zero = mk_smt_real("0.0");
      // B_dir(r) = eps_rel_dir * |r| + eps_abs
      smt_astt b_dir = mk_add(mk_mul(eps_rel_dir, abs_r), eps_abs);

      // Sign-dependent enclosure bounds via ITE:
      //   r >= 0: lower is computed, upper is exact (truncate-down shape)
      //   r <  0: lower is exact,    upper is computed (truncate-up shape)
      smt_astt r_nonneg = mk_le(zero, real_result); // r >= 0
      smt_astt ra_lo_expr =
        mk_ite(r_nonneg, mk_sub(real_result, b_dir), real_result);
      smt_astt ra_hi_expr =
        mk_ite(r_nonneg, real_result, mk_add(real_result, b_dir));

      // Introduce named enclosure variables. Use bidirectional inequalities to
      // survive Z3's solve-eqs tactic (same technique as the other tight paths).
      smt_astt ra_lo = mk_fresh(rs, "ra_lo_tz::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi_tz::", nullptr);

      // Pin ra_lo = ite(r >= 0, r - B_dir(r), r)
      assert_ast(mk_le(ra_lo, ra_lo_expr)); // ra_lo <= ra_lo_expr
      assert_ast(mk_le(ra_lo_expr, ra_lo)); // ra_lo_expr <= ra_lo

      // Pin ra_hi = ite(r >= 0, r, r + B_dir(r))
      assert_ast(mk_le(ra_hi, ra_hi_expr)); // ra_hi <= ra_hi_expr
      assert_ast(mk_le(ra_hi_expr, ra_hi)); // ra_hi_expr <= ra_hi

      // Containment: ra_lo <= result <= ra_hi
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_away(rounding_mode))
    {
      // Tight path for ROUND_TO_AWAY (round-to-nearest, ties away from zero).
      //
      // ROUND_TO_AWAY is a nearest-rounding mode: the rounding error is
      // bounded by the unit roundoff u = 1/2 * machine-epsilon, exactly as
      // for ROUND_TO_EVEN.  The enclosure is therefore symmetric:
      //   |fl_RTA(r) - r| <= eps_rel * |r| + eps_abs
      //
      // So:
      //   B(r)  = eps_rel * |r| + eps_abs    (same formula as nearest)
      //   ra_lo = r - B(r)
      //   ra_hi = r + B(r)
      //
      // The epsilon constants are the same as ROUND_TO_EVEN:
      //   eps_rel = 2^-53 (double) or 2^-24 (single) -- unit roundoff u
      //
      // NOTE: the Z3 numerator for eps_rel is the same as for the nearest
      // path (5551115123125783 for double, 5960464477539063 for single).
      // This is expected and correct -- the bound is identical.

      smt_sortt rs = mk_real_sort();
      smt_astt eps_rel = nullptr, eps_abs = nullptr;
      if (!select_nearest_eps(eps_rel, eps_abs))
      {
        // Unsupported format: fall back to unconstrained weak enclosure.
        return weak_enclosure_return("round-to-away: unsupported fp format");
      }

      smt_astt abs_r = abs_real(real_result);

      // B(r) = eps_rel * |r| + eps_abs
      smt_astt bound = mk_add(mk_mul(eps_rel, abs_r), eps_abs);
      smt_astt ra_lo_expr = mk_sub(real_result, bound);
      smt_astt ra_hi_expr = mk_add(real_result, bound);

      // Introduce named enclosure variables. Use bidirectional inequalities to
      // survive Z3's solve-eqs tactic (same technique as the other tight paths).
      smt_astt ra_lo = mk_fresh(rs, "ra_lo_aw::", nullptr);
      smt_astt ra_hi = mk_fresh(rs, "ra_hi_aw::", nullptr);

      // Pin ra_lo = r - B(r)
      assert_ast(mk_le(ra_lo, ra_lo_expr)); // ra_lo <= r - B(r)
      assert_ast(mk_le(ra_lo_expr, ra_lo)); // r - B(r) <= ra_lo

      // Pin ra_hi = r + B(r)
      assert_ast(mk_le(ra_hi, ra_hi_expr)); // ra_hi <= r + B(r)
      assert_ast(mk_le(ra_hi_expr, ra_hi)); // r + B(r) <= ra_hi

      // Containment: ra_lo <= result <= ra_hi
      assert_ast(mk_le(ra_lo, real_result));
      assert_ast(mk_le(real_result, ra_hi));
      assert_ast(mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else
    {
      // weak fallback: symbolic or unrecognised rounding mode
      return weak_enclosure_return("unrecognized symbolic rounding mode");
    }
  }
  else
  {
    unsigned int fraction_bits = fbv_type.fraction;
    unsigned int exponent_bits = fbv_type.exponent;

    auto double_spec = ieee_float_spect::double_precision();
    auto single_spec = ieee_float_spect::single_precision();

    smt_astt min_normal, min_subnormal, max_normal;

    // IEEE 754 double precision (64-bit): 11 exponent bits, 52 fraction bits
    if (exponent_bits == double_spec.e && fraction_bits == double_spec.f)
    {
      min_normal = get_double_min_normal();
      min_subnormal = get_double_min_subnormal();
      max_normal = get_double_max_normal();
    }
    // IEEE 754 single precision (32-bit): 8 exponent bits, 23 fraction bits
    else if (exponent_bits == single_spec.e && fraction_bits == single_spec.f)
    {
      min_normal = get_single_min_normal();
      min_subnormal = get_single_min_subnormal();
      max_normal = get_single_max_normal();
    }
    // Unsupported format - return original result
    else
    {
      log_warning(
        "Unsupported IEEE 754 format: exponent bits = {}, fraction bits = {}",
        exponent_bits,
        fraction_bits);
      return real_result;
    }

    smt_astt zero = mk_smt_real("0.0");

    // Get absolute value of result
    smt_astt abs_result =
      mk_ite(mk_lt(real_result, zero), mk_sub(zero, real_result), real_result);

    // Check for overflow
    smt_astt overflows = mk_gt(abs_result, max_normal);

    // Check for underflow to zero.
    // For nearest modes, values around the subnormal boundary should not be
    // forced to zero too aggressively; use 0.5 * min_subnormal threshold.
    smt_astt underflow_threshold = min_subnormal;
    smt_astt half_min_subnormal = mk_div(min_subnormal, mk_smt_real("2.0"));
    bool is_rne =
      smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode);
    bool is_rna = smt_fp_rounding_utils::is_round_to_away(rounding_mode);
    smt_astt underflows_to_zero;
    if (is_rne)
    {
      // ties-to-even: midpoint rounds to zero
      underflows_to_zero = mk_and(
        mk_le(abs_result, half_min_subnormal),
        mk_not(mk_eq(real_result, zero)));
    }
    else if (is_rna)
    {
      // ties-away: midpoint rounds away from zero
      underflows_to_zero = mk_and(
        mk_lt(abs_result, half_min_subnormal),
        mk_not(mk_eq(real_result, zero)));
    }
    else
    {
      underflows_to_zero = mk_and(
        mk_lt(abs_result, underflow_threshold),
        mk_not(mk_eq(real_result, zero)));
    }

    // If we have a special zero check (like for multiplication), use it
    if (operand_zero_check)
      underflows_to_zero =
        mk_and(underflows_to_zero, mk_not(operand_zero_check));

    // Check if result is in subnormal range
    smt_astt is_subnormal =
      mk_and(mk_ge(abs_result, min_subnormal), mk_lt(abs_result, min_normal));

    // For subnormal values, return the exact real result.
    // Subnormal arithmetic is exact when results are representable, and the
    // real arithmetic value already captures the correct value. No floor-based
    // quantization is applied here because ESBMC's SMT API has no floor/to_int
    // for reals; adding 0.5 without floor gives a wrong non-integer multiple.
    smt_astt subnormal_result = real_result;

    // Overflow result (approximate infinity)
    smt_astt overflow_result =
      mk_ite(mk_lt(real_result, zero), mk_sub(zero, max_normal), max_normal);

    // Apply IEEE 754 semantics: overflow > underflow > subnormal > normal
    smt_astt ieee_result = mk_ite(
      overflows,
      overflow_result,
      mk_ite(
        underflows_to_zero,
        zero,
        mk_ite(is_subnormal, subnormal_result, real_result)));

    // Handle special operand zero case for multiplication
    if (operand_zero_check)
      return mk_ite(operand_zero_check, zero, ieee_result);

    return ieee_result;
  }
}

smt_astt smt_solver_baset::get_double_max_normal()
{
  // IEEE 754 double: (2^53-1)*2^971 (exact rational, no rounding)
  static const std::string val =
    integer2string((power(2, 53) - 1) * power(2, 971));
  return mk_smt_real(val);
}

smt_astt smt_solver_baset::get_single_min_normal()
{
  // IEEE 754 single precision minimum normal positive value (2^-126)
  return mk_smt_real("1.1754943508222875e-38");
}

smt_astt smt_solver_baset::get_single_min_subnormal()
{
  // IEEE 754 single precision minimum positive subnormal value (2^-149)
  return mk_smt_real("1.4012984643248171e-45");
}

smt_astt smt_solver_baset::get_single_max_normal()
{
  // IEEE 754 single: (2^24-1)*2^104 (exact rational, no rounding)
  static const std::string val =
    integer2string((power(2, 24) - 1) * power(2, 104));
  return mk_smt_real(val);
}

smt_astt smt_solver_baset::get_double_inf_sentinel()
{
  // One above double max_normal: used to represent ±∞ in integer encoding.
  // This value satisfies |x| > max_normal, so isinf/isfinite predicates fire.
  static const std::string val =
    integer2string((power(2, 53) - 1) * power(2, 971) + 1);
  return mk_smt_real(val);
}

smt_astt smt_solver_baset::get_single_inf_sentinel()
{
  // One above single max_normal: used to represent ±∞ in integer encoding.
  static const std::string val =
    integer2string((power(2, 24) - 1) * power(2, 104) + 1);
  return mk_smt_real(val);
}

smt_astt smt_solver_baset::get_double_eps_rel()
{
  // Relative error bound for IEEE 754 double under round-to-nearest:
  // half machine epsilon = 2^-53; rounded UP at the last digit to ensure
  // the enclosure is conservative.
  return mk_smt_real("1.1102230246251566e-16");
}

smt_astt smt_solver_baset::get_single_eps_rel()
{
  // Relative error bound for IEEE 754 single under round-to-nearest:
  // half machine epsilon = 2^-24
  return mk_smt_real("5.960464477539063e-08");
}

smt_astt smt_solver_baset::get_double_eps_up()
{
  // B_dir relative error bound for IEEE 754 double under round-toward-+inf:
  // eps_rel_dir = 2^-52 (the full machine epsilon for double, DBL_EPSILON).
  // Conservative decimal: strictly greater than 2^-52 to preserve soundness.
  return mk_smt_real("2.2204460492503131e-16");
}

smt_astt smt_solver_baset::get_single_eps_up()
{
  // B_dir relative error bound for IEEE 754 single under round-toward-+inf:
  // eps_rel_dir = 2^-23 (the full machine epsilon for single, FLT_EPSILON).
  // Exact decimal representation of 2^-23; no rounding required.
  return mk_smt_real("1.1920928955078125e-07");
}

smt_astt smt_solver_baset::convert_is_nan(const expr2tc &expr)
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
static smt_astt
get_max_normal(smt_solver_baset &conv, const floatbv_type2t &fbv_type)
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
static smt_astt
get_min_normal(smt_solver_baset &conv, const floatbv_type2t &fbv_type)
{
  const auto single_spec = ieee_float_spect::single_precision();
  const auto double_spec = ieee_float_spect::double_precision();
  if (fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
    return conv.get_single_min_normal();
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
    return conv.get_double_min_normal();
  return nullptr;
}

smt_astt smt_solver_baset::convert_is_inf(const expr2tc &expr)
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

smt_astt smt_solver_baset::convert_is_normal(const expr2tc &expr)
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

smt_astt smt_solver_baset::convert_is_finite(const expr2tc &expr)
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

smt_astt smt_solver_baset::convert_signbit(const expr2tc &expr)
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

smt_astt smt_solver_baset::convert_rounding_mode(const expr2tc &expr)
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
