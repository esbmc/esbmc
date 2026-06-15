#include <solvers/smt/fp/ir_ieee_conv.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_fp_rounding_utils.h>
#include <irep2/irep2_expr.h>
#include <util/arith_tools.h>
#include <util/message.h>
#include <util/message/format.h>

ir_ieee_convt::ir_ieee_convt(smt_convt *_ctx) : ctx(_ctx)
{
}

void ir_ieee_convt::propagate_interval(smt_astt lhs, smt_astt rhs)
{
  auto it = ir_ra_interval_map.find(rhs);
  if (it != ir_ra_interval_map.end())
    ir_ra_interval_map[lhs] = it->second;
}

void ir_ieee_convt::assert_symbol_range(
  const std::string &name,
  smt_astt sym_ast,
  const symbol2t &sym)
{
  if (
    !ctx->ir_ieee || !ctx->int_encoding ||
    (!is_unsignedbv_type(sym.type) && !is_signedbv_type(sym.type)) ||
    sym.type->get_width() >= 32 || !ir_ieee_ranged_syms.insert(name).second)
    return;

  const unsigned w = sym.type->get_width();
  if (is_unsignedbv_type(sym.type))
  {
    ctx->assert_ast(ctx->mk_le(ctx->mk_smt_int(BigInt(0)), sym_ast));
    ctx->assert_ast(
      ctx->mk_le(sym_ast, ctx->mk_smt_int(BigInt::power2(w) - 1)));
  }
  else
  {
    ctx->assert_ast(
      ctx->mk_le(ctx->mk_smt_int(-BigInt::power2(w - 1)), sym_ast));
    ctx->assert_ast(
      ctx->mk_le(sym_ast, ctx->mk_smt_int(BigInt::power2(w - 1) - 1)));
  }
}

ir_ieee_convt::ra_interval_t ir_ieee_convt::get_iv(smt_astt t) const
{
  auto it = ir_ra_interval_map.find(t);
  return it != ir_ra_interval_map.end() ? it->second : ra_interval_t{t, t};
}

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type,
  const expr2tc &rounding_mode)
{
  if (smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode))
    return apply_ieee754_rne_enclosure(real_result, lo_r, hi_r, fbv_type);
  if (smt_fp_rounding_utils::is_round_to_away(rounding_mode))
    return apply_ieee754_rna_enclosure(real_result, lo_r, hi_r, fbv_type);
  if (smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode))
    return apply_ieee754_rup_enclosure(real_result, lo_r, hi_r, fbv_type);
  if (smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode))
    return apply_ieee754_rdn_enclosure(real_result, lo_r, hi_r, fbv_type);
  return apply_ieee754_rtz_enclosure(real_result, lo_r, hi_r, fbv_type);
}

// ---------------------------------------------------------------------------
// Enclosure helpers
// ---------------------------------------------------------------------------

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_ieee754_rne_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  smt_astt eps_rel = nullptr, eps_abs = nullptr;
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
  {
    eps_rel = ctx->get_double_eps_rel();
    eps_abs = ctx->get_double_min_subnormal();
  }
  else if (
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
  {
    eps_rel = ctx->get_single_eps_rel();
    eps_abs = ctx->get_single_min_subnormal();
  }
  else
  {
    assert(!"apply_ieee754_rne_enclosure: unsupported FP format");
    abort();
  }

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");

  smt_astt abs_lo =
    ctx->mk_ite(ctx->mk_lt(lo_r, zero), ctx->mk_sub(zero, lo_r), lo_r);
  smt_astt bound_lo = ctx->mk_add(ctx->mk_mul(eps_rel, abs_lo), eps_abs);

  smt_astt abs_hi =
    ctx->mk_ite(ctx->mk_lt(hi_r, zero), ctx->mk_sub(zero, hi_r), hi_r);
  smt_astt bound_hi = ctx->mk_add(ctx->mk_mul(eps_rel, abs_hi), eps_abs);

  smt_astt ra_lo_expr = ctx->mk_sub(lo_r, bound_lo);
  smt_astt ra_hi_expr = ctx->mk_add(hi_r, bound_hi);

  smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo::", nullptr);
  smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi::", nullptr);

  ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
  ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));
  ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
  ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

  ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
  ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
  ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

  return {ra_lo, ra_hi};
}

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_ieee754_rna_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  smt_astt eps_rel = nullptr, eps_abs = nullptr;
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
  {
    eps_rel = ctx->get_double_eps_rel();
    eps_abs = ctx->get_double_min_subnormal();
  }
  else if (
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
  {
    eps_rel = ctx->get_single_eps_rel();
    eps_abs = ctx->get_single_min_subnormal();
  }
  else
  {
    assert(!"apply_ieee754_rna_enclosure: unsupported FP format");
    abort();
  }

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");

  smt_astt abs_lo =
    ctx->mk_ite(ctx->mk_lt(lo_r, zero), ctx->mk_sub(zero, lo_r), lo_r);
  smt_astt bound_lo = ctx->mk_add(ctx->mk_mul(eps_rel, abs_lo), eps_abs);

  smt_astt abs_hi =
    ctx->mk_ite(ctx->mk_lt(hi_r, zero), ctx->mk_sub(zero, hi_r), hi_r);
  smt_astt bound_hi = ctx->mk_add(ctx->mk_mul(eps_rel, abs_hi), eps_abs);

  smt_astt ra_lo_expr = ctx->mk_sub(lo_r, bound_lo);
  smt_astt ra_hi_expr = ctx->mk_add(hi_r, bound_hi);

  smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_aw::", nullptr);
  smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_aw::", nullptr);

  ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
  ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));
  ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
  ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

  ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
  ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
  ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

  return {ra_lo, ra_hi};
}

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_ieee754_rup_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
  {
    eps_rel_dir = ctx->get_double_eps_up();
    eps_abs = ctx->get_double_min_subnormal();
  }
  else if (
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
  {
    eps_rel_dir = ctx->get_single_eps_up();
    eps_abs = ctx->get_single_min_subnormal();
  }
  else
  {
    assert(!"apply_ieee754_rup_enclosure: unsupported FP format");
    abort();
  }

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");

  smt_astt abs_hi =
    ctx->mk_ite(ctx->mk_lt(hi_r, zero), ctx->mk_sub(zero, hi_r), hi_r);
  smt_astt bound_hi = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_hi), eps_abs);
  smt_astt ra_hi_expr = ctx->mk_add(hi_r, bound_hi);

  smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_up::", nullptr);
  smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_up::", nullptr);

  ctx->assert_ast(ctx->mk_le(ra_lo, lo_r));
  ctx->assert_ast(ctx->mk_le(lo_r, ra_lo));

  ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
  ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

  ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
  ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
  ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

  return {ra_lo, ra_hi};
}

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_ieee754_rdn_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
  {
    eps_rel_dir = ctx->get_double_eps_up();
    eps_abs = ctx->get_double_min_subnormal();
  }
  else if (
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
  {
    eps_rel_dir = ctx->get_single_eps_up();
    eps_abs = ctx->get_single_min_subnormal();
  }
  else
  {
    assert(!"apply_ieee754_rdn_enclosure: unsupported FP format");
    abort();
  }

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");

  smt_astt abs_lo =
    ctx->mk_ite(ctx->mk_lt(lo_r, zero), ctx->mk_sub(zero, lo_r), lo_r);
  smt_astt bound_lo = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_lo), eps_abs);
  smt_astt ra_lo_expr = ctx->mk_sub(lo_r, bound_lo);

  smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_dn::", nullptr);
  smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_dn::", nullptr);

  ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
  ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));

  ctx->assert_ast(ctx->mk_le(ra_hi, hi_r));
  ctx->assert_ast(ctx->mk_le(hi_r, ra_hi));

  ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
  ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
  ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

  return {ra_lo, ra_hi};
}

std::pair<smt_astt, smt_astt> ir_ieee_convt::apply_ieee754_rtz_enclosure(
  smt_astt real_result,
  smt_astt lo_r,
  smt_astt hi_r,
  const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
  if (fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f)
  {
    eps_rel_dir = ctx->get_double_eps_up();
    eps_abs = ctx->get_double_min_subnormal();
  }
  else if (
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f)
  {
    eps_rel_dir = ctx->get_single_eps_up();
    eps_abs = ctx->get_single_min_subnormal();
  }
  else
  {
    assert(!"apply_ieee754_rtz_enclosure: unsupported FP format");
    abort();
  }

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");

  smt_astt abs_lo =
    ctx->mk_ite(ctx->mk_lt(lo_r, zero), ctx->mk_sub(zero, lo_r), lo_r);
  smt_astt abs_hi =
    ctx->mk_ite(ctx->mk_lt(hi_r, zero), ctx->mk_sub(zero, hi_r), hi_r);

  smt_astt bound_lo = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_lo), eps_abs);
  smt_astt bound_hi = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_hi), eps_abs);

  smt_astt abs_max = ctx->mk_ite(ctx->mk_le(abs_lo, abs_hi), abs_hi, abs_lo);
  smt_astt bound_max = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_max), eps_abs);

  smt_astt lo_nonneg = ctx->mk_le(zero, lo_r);
  smt_astt hi_nonpos = ctx->mk_le(hi_r, zero);

  smt_astt ra_lo_expr = ctx->mk_ite(
    lo_nonneg,
    ctx->mk_sub(lo_r, bound_lo),
    ctx->mk_ite(hi_nonpos, lo_r, ctx->mk_sub(lo_r, bound_max)));

  smt_astt ra_hi_expr = ctx->mk_ite(
    lo_nonneg,
    hi_r,
    ctx->mk_ite(
      hi_nonpos, ctx->mk_add(hi_r, bound_hi), ctx->mk_add(hi_r, bound_max)));

  smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_tz::", nullptr);
  smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_tz::", nullptr);

  ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
  ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));

  ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
  ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

  ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
  ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
  ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

  return {ra_lo, ra_hi};
}

// ---------------------------------------------------------------------------
// apply_ieee754_semantics
// ---------------------------------------------------------------------------

smt_astt ir_ieee_convt::apply_ieee754_semantics(
  smt_astt real_result,
  const floatbv_type2t &fbv_type,
  smt_astt operand_zero_check,
  const expr2tc &rounding_mode)
{
  if (ctx->options.get_bool_option("ir-ieee"))
  {
    auto weak_enclosure_return =
      [this, &real_result, &fbv_type](const char *reason) -> smt_astt {
      log_warning(
        "using weak IEEE754 enclosure ({}), exp={}, frac={}",
        reason,
        fbv_type.exponent,
        fbv_type.fraction);
      smt_sortt rs = ctx->mk_real_sort();
      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_weak::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_weak::", nullptr);
      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));
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
        eps_rel = ctx->get_double_eps_rel();
        eps_abs = ctx->get_double_min_subnormal();
        return true;
      }
      if (
        fbv_type.exponent == single_spec.e &&
        fbv_type.fraction == single_spec.f)
      {
        eps_rel = ctx->get_single_eps_rel();
        eps_abs = ctx->get_single_min_subnormal();
        return true;
      }
      return false;
    };

    auto abs_real = [this](smt_astt r) -> smt_astt {
      smt_astt zero = ctx->mk_smt_real("0.0");
      return ctx->mk_ite(ctx->mk_lt(r, zero), ctx->mk_sub(zero, r), r);
    };

    auto select_directed_eps =
      [this, &fbv_type](smt_astt &eps_rel_dir, smt_astt &eps_abs) -> bool {
      const auto double_spec = ieee_float_spect::double_precision();
      const auto single_spec = ieee_float_spect::single_precision();
      if (
        fbv_type.exponent == double_spec.e &&
        fbv_type.fraction == double_spec.f)
      {
        eps_rel_dir = ctx->get_double_eps_up();
        eps_abs = ctx->get_double_min_subnormal();
        return true;
      }
      if (
        fbv_type.exponent == single_spec.e &&
        fbv_type.fraction == single_spec.f)
      {
        eps_rel_dir = ctx->get_single_eps_up();
        eps_abs = ctx->get_single_min_subnormal();
        return true;
      }
      return false;
    };

    if (smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode))
    {
      smt_astt eps_rel = nullptr, eps_abs = nullptr;
      if (!select_nearest_eps(eps_rel, eps_abs))
        return weak_enclosure_return("nearest: unsupported fp format");

      smt_sortt rs = ctx->mk_real_sort();
      smt_astt abs_r = abs_real(real_result);

      smt_astt bound = ctx->mk_add(ctx->mk_mul(eps_rel, abs_r), eps_abs);

      smt_astt ra_lo_expr = ctx->mk_sub(real_result, bound);
      smt_astt ra_hi_expr = ctx->mk_add(real_result, bound);

      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi::", nullptr);
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
      ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));
      ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
      ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode))
    {
      smt_sortt rs = ctx->mk_real_sort();
      smt_astt eps_up, eps_abs;
      if (!select_directed_eps(eps_up, eps_abs))
        return weak_enclosure_return(
          "round-to-plus-inf: unsupported fp format");

      smt_astt abs_r = abs_real(real_result);
      smt_astt b_dir = ctx->mk_add(ctx->mk_mul(eps_up, abs_r), eps_abs);
      smt_astt ra_hi_expr = ctx->mk_add(real_result, b_dir);

      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_up::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_up::", nullptr);

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_lo));

      ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
      ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode))
    {
      smt_sortt rs = ctx->mk_real_sort();
      smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
      if (!select_directed_eps(eps_rel_dir, eps_abs))
        return weak_enclosure_return(
          "round-to-minus-inf: unsupported fp format");

      smt_astt abs_r = abs_real(real_result);
      smt_astt b_dir = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_r), eps_abs);
      smt_astt ra_lo_expr = ctx->mk_sub(real_result, b_dir);

      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_dn::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_dn::", nullptr);

      ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
      ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));

      ctx->assert_ast(ctx->mk_le(ra_hi, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_zero(rounding_mode))
    {
      smt_sortt rs = ctx->mk_real_sort();
      smt_astt eps_rel_dir = nullptr, eps_abs = nullptr;
      if (!select_directed_eps(eps_rel_dir, eps_abs))
        return weak_enclosure_return("round-to-zero: unsupported fp format");

      smt_astt abs_r = abs_real(real_result);
      smt_astt zero = ctx->mk_smt_real("0.0");
      smt_astt b_dir = ctx->mk_add(ctx->mk_mul(eps_rel_dir, abs_r), eps_abs);

      smt_astt r_nonneg = ctx->mk_le(zero, real_result);
      smt_astt ra_lo_expr =
        ctx->mk_ite(r_nonneg, ctx->mk_sub(real_result, b_dir), real_result);
      smt_astt ra_hi_expr =
        ctx->mk_ite(r_nonneg, real_result, ctx->mk_add(real_result, b_dir));

      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_tz::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_tz::", nullptr);

      ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
      ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));

      ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
      ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else if (smt_fp_rounding_utils::is_round_to_away(rounding_mode))
    {
      smt_sortt rs = ctx->mk_real_sort();
      smt_astt eps_rel = nullptr, eps_abs = nullptr;
      if (!select_nearest_eps(eps_rel, eps_abs))
        return weak_enclosure_return("round-to-away: unsupported fp format");

      smt_astt abs_r = abs_real(real_result);
      smt_astt bound = ctx->mk_add(ctx->mk_mul(eps_rel, abs_r), eps_abs);
      smt_astt ra_lo_expr = ctx->mk_sub(real_result, bound);
      smt_astt ra_hi_expr = ctx->mk_add(real_result, bound);

      smt_astt ra_lo = ctx->mk_fresh(rs, "ra_lo_aw::", nullptr);
      smt_astt ra_hi = ctx->mk_fresh(rs, "ra_hi_aw::", nullptr);

      ctx->assert_ast(ctx->mk_le(ra_lo, ra_lo_expr));
      ctx->assert_ast(ctx->mk_le(ra_lo_expr, ra_lo));

      ctx->assert_ast(ctx->mk_le(ra_hi, ra_hi_expr));
      ctx->assert_ast(ctx->mk_le(ra_hi_expr, ra_hi));

      ctx->assert_ast(ctx->mk_le(ra_lo, real_result));
      ctx->assert_ast(ctx->mk_le(real_result, ra_hi));
      ctx->assert_ast(ctx->mk_le(ra_lo, ra_hi));

      return real_result;
    }
    else
    {
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

    if (exponent_bits == double_spec.e && fraction_bits == double_spec.f)
    {
      min_normal = ctx->get_double_min_normal();
      min_subnormal = ctx->get_double_min_subnormal();
      max_normal = ctx->get_double_max_normal();
    }
    else if (exponent_bits == single_spec.e && fraction_bits == single_spec.f)
    {
      min_normal = ctx->get_single_min_normal();
      min_subnormal = ctx->get_single_min_subnormal();
      max_normal = ctx->get_single_max_normal();
    }
    else
    {
      log_warning(
        "Unsupported IEEE 754 format: exponent bits = {}, fraction bits = {}",
        exponent_bits,
        fraction_bits);
      return real_result;
    }

    smt_astt zero = ctx->mk_smt_real("0.0");

    smt_astt abs_result = ctx->mk_ite(
      ctx->mk_lt(real_result, zero),
      ctx->mk_sub(zero, real_result),
      real_result);

    smt_astt overflows = ctx->mk_gt(abs_result, max_normal);

    smt_astt underflow_threshold = min_subnormal;
    smt_astt half_min_subnormal =
      ctx->mk_div(min_subnormal, ctx->mk_smt_real("2.0"));
    bool is_rne =
      smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode);
    bool is_rna = smt_fp_rounding_utils::is_round_to_away(rounding_mode);
    smt_astt underflows_to_zero;
    if (is_rne)
    {
      underflows_to_zero = ctx->mk_and(
        ctx->mk_le(abs_result, half_min_subnormal),
        ctx->mk_not(ctx->mk_eq(real_result, zero)));
    }
    else if (is_rna)
    {
      underflows_to_zero = ctx->mk_and(
        ctx->mk_lt(abs_result, half_min_subnormal),
        ctx->mk_not(ctx->mk_eq(real_result, zero)));
    }
    else
    {
      underflows_to_zero = ctx->mk_and(
        ctx->mk_lt(abs_result, underflow_threshold),
        ctx->mk_not(ctx->mk_eq(real_result, zero)));
    }

    if (operand_zero_check)
      underflows_to_zero =
        ctx->mk_and(underflows_to_zero, ctx->mk_not(operand_zero_check));

    smt_astt is_subnormal = ctx->mk_and(
      ctx->mk_ge(abs_result, min_subnormal),
      ctx->mk_lt(abs_result, min_normal));

    smt_astt subnormal_result = real_result;

    smt_astt overflow_result = ctx->mk_ite(
      ctx->mk_lt(real_result, zero), ctx->mk_sub(zero, max_normal), max_normal);

    smt_astt ieee_result = ctx->mk_ite(
      overflows,
      overflow_result,
      ctx->mk_ite(
        underflows_to_zero,
        zero,
        ctx->mk_ite(is_subnormal, subnormal_result, real_result)));

    if (operand_zero_check)
      return ctx->mk_ite(operand_zero_check, zero, ieee_result);

    return ieee_result;
  }
}

// ---------------------------------------------------------------------------
// Interval-lifted encoding of the six IEEE operations
// ---------------------------------------------------------------------------

// Returns true if the format is single or double precision.
static bool is_known_fp_format(const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  return (
    (fbv_type.exponent == double_spec.e &&
     fbv_type.fraction == double_spec.f) ||
    (fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f));
}

// Returns true if the rounding mode can be interval-lifted.
static bool is_liftable_rounding_mode(const expr2tc &rm)
{
  return smt_fp_rounding_utils::is_nearest_rounding_mode(rm) ||
         smt_fp_rounding_utils::is_round_to_away(rm) ||
         smt_fp_rounding_utils::is_round_to_plus_inf(rm) ||
         smt_fp_rounding_utils::is_round_to_minus_inf(rm) ||
         smt_fp_rounding_utils::is_round_to_zero(rm);
}

smt_astt ir_ieee_convt::encode_ieee_add(const expr2tc &expr)
{
  smt_astt side1 = ctx->convert_ast(to_ieee_add2t(expr).side_1);
  smt_astt side2 = ctx->convert_ast(to_ieee_add2t(expr).side_2);
  smt_astt real_result = ctx->mk_add(side1, side2);
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  const expr2tc &rounding_mode = to_ieee_add2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_liftable_rounding_mode(rounding_mode) && is_known_fp_format(fbv_type))
  {
    ra_interval_t iv1 = get_iv(side1);
    ra_interval_t iv2 = get_iv(side2);
    smt_astt lo_r = ctx->mk_add(iv1.lo, iv2.lo);
    smt_astt hi_r = ctx->mk_add(iv1.hi, iv2.hi);
    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    ir_ra_interval_map[real_result] = {bounds.first, bounds.second};
    return real_result;
  }
  return apply_ieee754_semantics(real_result, fbv_type, nullptr, rounding_mode);
}

smt_astt ir_ieee_convt::encode_ieee_sub(const expr2tc &expr)
{
  smt_astt side1 = ctx->convert_ast(to_ieee_sub2t(expr).side_1);
  smt_astt side2 = ctx->convert_ast(to_ieee_sub2t(expr).side_2);
  smt_astt real_result = ctx->mk_sub(side1, side2);
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  const expr2tc &rounding_mode = to_ieee_sub2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_liftable_rounding_mode(rounding_mode) && is_known_fp_format(fbv_type))
  {
    ra_interval_t iv1 = get_iv(side1);
    ra_interval_t iv2 = get_iv(side2);
    smt_astt lo_r = ctx->mk_sub(iv1.lo, iv2.hi); // L_R = L_x - U_y
    smt_astt hi_r = ctx->mk_sub(iv1.hi, iv2.lo); // U_R = U_x - L_y
    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    ir_ra_interval_map[real_result] = {bounds.first, bounds.second};
    return real_result;
  }
  return apply_ieee754_semantics(real_result, fbv_type, nullptr, rounding_mode);
}

smt_astt ir_ieee_convt::encode_ieee_mul(const expr2tc &expr)
{
  smt_astt side1 = ctx->convert_ast(to_ieee_mul2t(expr).side_1);
  smt_astt side2 = ctx->convert_ast(to_ieee_mul2t(expr).side_2);
  smt_astt real_result = ctx->mk_mul(side1, side2);
  smt_astt zero = ctx->mk_smt_real("0.0");
  smt_astt operand_is_zero =
    ctx->mk_or(ctx->mk_eq(side1, zero), ctx->mk_eq(side2, zero));
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  const expr2tc &rounding_mode = to_ieee_mul2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_liftable_rounding_mode(rounding_mode) && is_known_fp_format(fbv_type))
  {
    ra_interval_t iv1 = get_iv(side1);
    ra_interval_t iv2 = get_iv(side2);
    smt_astt p1 = ctx->mk_mul(iv1.lo, iv2.lo);
    smt_astt p2 = ctx->mk_mul(iv1.lo, iv2.hi);
    smt_astt p3 = ctx->mk_mul(iv1.hi, iv2.lo);
    smt_astt p4 = ctx->mk_mul(iv1.hi, iv2.hi);
    smt_astt lo_r = ctx->mk_ite(
      ctx->mk_le(p1, p2),
      ctx->mk_ite(
        ctx->mk_le(p1, p3),
        ctx->mk_ite(ctx->mk_le(p1, p4), p1, p4),
        ctx->mk_ite(ctx->mk_le(p3, p4), p3, p4)),
      ctx->mk_ite(
        ctx->mk_le(p2, p3),
        ctx->mk_ite(ctx->mk_le(p2, p4), p2, p4),
        ctx->mk_ite(ctx->mk_le(p3, p4), p3, p4)));
    smt_astt hi_r = ctx->mk_ite(
      ctx->mk_le(p2, p1),
      ctx->mk_ite(
        ctx->mk_le(p3, p1),
        ctx->mk_ite(ctx->mk_le(p4, p1), p1, p4),
        ctx->mk_ite(ctx->mk_le(p4, p3), p3, p4)),
      ctx->mk_ite(
        ctx->mk_le(p3, p2),
        ctx->mk_ite(ctx->mk_le(p4, p2), p2, p4),
        ctx->mk_ite(ctx->mk_le(p4, p3), p3, p4)));
    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    ir_ra_interval_map[real_result] = {bounds.first, bounds.second};
    return real_result;
  }
  return apply_ieee754_semantics(
    real_result, fbv_type, operand_is_zero, rounding_mode);
}

smt_astt ir_ieee_convt::encode_ieee_div(const expr2tc &expr)
{
  smt_astt side1 = ctx->convert_ast(to_ieee_div2t(expr).side_1);
  smt_astt side2 = ctx->convert_ast(to_ieee_div2t(expr).side_2);
  smt_astt zero = ctx->get_zero_real();
  smt_astt div_by_zero = ctx->mk_eq(side2, zero);
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  const auto single_spec = ieee_float_spect::single_precision();
  const auto double_spec = ieee_float_spect::double_precision();
  const bool is_single =
    fbv_type.exponent == single_spec.e && fbv_type.fraction == single_spec.f;
  const bool is_double =
    fbv_type.exponent == double_spec.e && fbv_type.fraction == double_spec.f;

  if (!is_single && !is_double)
  {
    smt_astt real_result = ctx->mk_div(side1, side2);
    return apply_ieee754_semantics(
      real_result, fbv_type, nullptr, to_ieee_div2t(expr).rounding_mode);
  }

  smt_astt sentinel = ctx->get_double_inf_sentinel();
  smt_astt inf_result =
    ctx->mk_ite(ctx->mk_lt(side1, zero), ctx->mk_sub(zero, sentinel), sentinel);
  smt_astt real_result = ctx->mk_div(side1, side2);
  const expr2tc &rounding_mode = to_ieee_div2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_liftable_rounding_mode(rounding_mode))
  {
    ra_interval_t iv1 = get_iv(side1);
    ra_interval_t iv2 = get_iv(side2);

    smt_astt denom_admissible =
      ctx->mk_or(ctx->mk_lt(zero, iv2.lo), ctx->mk_lt(iv2.hi, zero));

    smt_astt q1 = ctx->mk_div(iv1.lo, iv2.lo);
    smt_astt q2 = ctx->mk_div(iv1.lo, iv2.hi);
    smt_astt q3 = ctx->mk_div(iv1.hi, iv2.lo);
    smt_astt q4 = ctx->mk_div(iv1.hi, iv2.hi);
    smt_astt lo_r_full = ctx->mk_ite(
      ctx->mk_le(q1, q2),
      ctx->mk_ite(
        ctx->mk_le(q1, q3),
        ctx->mk_ite(ctx->mk_le(q1, q4), q1, q4),
        ctx->mk_ite(ctx->mk_le(q3, q4), q3, q4)),
      ctx->mk_ite(
        ctx->mk_le(q2, q3),
        ctx->mk_ite(ctx->mk_le(q2, q4), q2, q4),
        ctx->mk_ite(ctx->mk_le(q3, q4), q3, q4)));
    smt_astt hi_r_full = ctx->mk_ite(
      ctx->mk_le(q2, q1),
      ctx->mk_ite(
        ctx->mk_le(q3, q1),
        ctx->mk_ite(ctx->mk_le(q4, q1), q1, q4),
        ctx->mk_ite(ctx->mk_le(q4, q3), q3, q4)),
      ctx->mk_ite(
        ctx->mk_le(q3, q2),
        ctx->mk_ite(ctx->mk_le(q4, q2), q2, q4),
        ctx->mk_ite(ctx->mk_le(q4, q3), q3, q4)));

    smt_astt d_lo = ctx->mk_div(iv1.lo, side2);
    smt_astt d_hi = ctx->mk_div(iv1.hi, side2);
    smt_astt lo_r_point = ctx->mk_ite(ctx->mk_le(d_lo, d_hi), d_lo, d_hi);
    smt_astt hi_r_point = ctx->mk_ite(ctx->mk_le(d_hi, d_lo), d_lo, d_hi);

    smt_astt lo_r = ctx->mk_ite(denom_admissible, lo_r_full, lo_r_point);
    smt_astt hi_r = ctx->mk_ite(denom_admissible, hi_r_full, hi_r_point);

    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    smt_astt result = ctx->mk_ite(div_by_zero, inf_result, real_result);
    ir_ra_interval_map[result] = {bounds.first, bounds.second};
    return result;
  }

  smt_astt ieee_result =
    apply_ieee754_semantics(real_result, fbv_type, nullptr, rounding_mode);
  return ctx->mk_ite(div_by_zero, inf_result, ieee_result);
}

smt_astt ir_ieee_convt::encode_ieee_fma(const expr2tc &expr)
{
  smt_astt val1 = ctx->convert_ast(to_ieee_fma2t(expr).value_1);
  smt_astt val2 = ctx->convert_ast(to_ieee_fma2t(expr).value_2);
  smt_astt val3 = ctx->convert_ast(to_ieee_fma2t(expr).value_3);
  smt_astt intermediate = ctx->mk_mul(val1, val2);
  smt_astt real_result = ctx->mk_add(intermediate, val3);
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  return apply_ieee754_semantics(
    real_result, fbv_type, nullptr, to_ieee_fma2t(expr).rounding_mode);
}

smt_astt ir_ieee_convt::encode_ieee_sqrt(const expr2tc &expr)
{
  smt_astt operand = ctx->convert_ast(to_ieee_sqrt2t(expr).value);
  const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
  const expr2tc &rounding_mode = to_ieee_sqrt2t(expr).rounding_mode;

  smt_sortt rs = ctx->mk_real_sort();
  smt_astt zero = ctx->mk_smt_real("0.0");
  smt_astt op_nonneg = ctx->mk_le(zero, operand);

  smt_astt sqrt_pos = ctx->mk_fresh(rs, "ra_sqrt::", nullptr);
  ctx->assert_ast(
    ctx->mk_or(ctx->mk_not(op_nonneg), ctx->mk_le(zero, sqrt_pos)));
  ctx->assert_ast(ctx->mk_or(
    ctx->mk_not(op_nonneg),
    ctx->mk_eq(ctx->mk_mul(sqrt_pos, sqrt_pos), operand)));

  smt_astt sqrt_nan = ctx->mk_fresh(rs, "ra_sqrt_nan::", nullptr);

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_liftable_rounding_mode(rounding_mode) && is_known_fp_format(fbv_type))
  {
    ra_interval_t iv = get_iv(operand);

    smt_astt iv_lo_pos = ctx->mk_ite(ctx->mk_lt(iv.lo, zero), zero, iv.lo);
    smt_astt iv_hi_pos = ctx->mk_ite(ctx->mk_lt(iv.hi, zero), zero, iv.hi);

    smt_astt lo_r = ctx->mk_fresh(rs, "ra_sqrt_lo::", nullptr);
    smt_astt hi_r = ctx->mk_fresh(rs, "ra_sqrt_hi::", nullptr);
    ctx->assert_ast(ctx->mk_le(zero, lo_r));
    ctx->assert_ast(ctx->mk_eq(ctx->mk_mul(lo_r, lo_r), iv_lo_pos));
    ctx->assert_ast(ctx->mk_le(zero, hi_r));
    ctx->assert_ast(ctx->mk_eq(ctx->mk_mul(hi_r, hi_r), iv_hi_pos));

    auto bounds =
      apply_enclosure(sqrt_pos, lo_r, hi_r, fbv_type, rounding_mode);

    smt_astt sqrt_result = ctx->mk_ite(op_nonneg, sqrt_pos, sqrt_nan);
    smt_astt map_lo = ctx->mk_ite(op_nonneg, bounds.first, sqrt_result);
    smt_astt map_hi = ctx->mk_ite(op_nonneg, bounds.second, sqrt_result);
    ir_ra_interval_map[sqrt_result] = {map_lo, map_hi};
    return sqrt_result;
  }

  smt_astt pos_result =
    apply_ieee754_semantics(sqrt_pos, fbv_type, nullptr, rounding_mode);
  return ctx->mk_ite(op_nonneg, pos_result, sqrt_nan);
}
