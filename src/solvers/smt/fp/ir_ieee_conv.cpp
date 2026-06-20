#include <solvers/smt/fp/ir_ieee_conv.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/smt_fp_rounding_utils.h>
#include <irep2/irep2_type.h>
#include <util/ieee_float.h>

ir_ieee_convt::ir_ieee_convt(smt_solver_baset *ctx) : ctx(ctx)
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

ir_ieee_convt::ra_interval_t ir_ieee_convt::get_interval(smt_astt t) const
{
  auto it = ir_ra_interval_map.find(t);
  return it != ir_ra_interval_map.end() ? it->second : ra_interval_t{t, t};
}

void ir_ieee_convt::store_interval(smt_astt t, smt_astt lo, smt_astt hi)
{
  ir_ra_interval_map[t] = {lo, hi};
}

void ir_ieee_convt::store_nan_pred(smt_astt t, smt_astt nan_pred)
{
  ir_ieee_nan_map[t] = nan_pred;
}

smt_astt ir_ieee_convt::get_nan_pred(smt_astt t) const
{
  auto it = ir_ieee_nan_map.find(t);
  return it != ir_ieee_nan_map.end() ? it->second : nullptr;
}

void ir_ieee_convt::propagate_nan_pred(smt_astt lhs, smt_astt rhs)
{
  auto it = ir_ieee_nan_map.find(rhs);
  if (it != ir_ieee_nan_map.end())
    ir_ieee_nan_map[lhs] = it->second;
}

smt_astt ir_ieee_convt::combine_nan_preds(smt_astt a, smt_astt b) const
{
  if (!a && !b)
    return nullptr;
  if (!a)
    return b;
  if (!b)
    return a;
  return ctx->mk_or(a, b);
}

void ir_ieee_convt::store_combined_nan_pred(
  smt_astt result, smt_astt s1, smt_astt s2)
{
  smt_astt nan_p = combine_nan_preds(get_nan_pred(s1), get_nan_pred(s2));
  if (nan_p)
    store_nan_pred(result, nan_p);
}

smt_astt
ir_ieee_convt::apply_nan_cmp(smt_astt cmp, smt_astt a, smt_astt b, bool is_neq)
{
  smt_astt either_nan = combine_nan_preds(get_nan_pred(a), get_nan_pred(b));
  if (!either_nan)
    return cmp;
  return ctx->mk_ite(either_nan, ctx->mk_smt_bool(is_neq), cmp);
}

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

static bool is_known_rounding_mode(const expr2tc &rounding_mode)
{
  return smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_away(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_zero(rounding_mode);
}

static bool is_single_or_double(const floatbv_type2t &fbv_type)
{
  const auto double_spec = ieee_float_spect::double_precision();
  const auto single_spec = ieee_float_spect::single_precision();
  return (fbv_type.exponent == double_spec.e &&
          fbv_type.fraction == double_spec.f) ||
         (fbv_type.exponent == single_spec.e &&
          fbv_type.fraction == single_spec.f);
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
    is_known_rounding_mode(rounding_mode) && is_single_or_double(fbv_type))
  {
    ra_interval_t iv1 = get_interval(side1);
    ra_interval_t iv2 = get_interval(side2);
    smt_astt lo_r = ctx->mk_add(iv1.lo, iv2.lo);
    smt_astt hi_r = ctx->mk_add(iv1.hi, iv2.hi);
    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    store_interval(real_result, bounds.first, bounds.second);
    store_combined_nan_pred(real_result, side1, side2);
    return real_result;
  }
  return ctx->apply_ieee754_semantics(
    real_result, fbv_type, nullptr, rounding_mode);
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
    is_known_rounding_mode(rounding_mode) && is_single_or_double(fbv_type))
  {
    ra_interval_t iv1 = get_interval(side1);
    ra_interval_t iv2 = get_interval(side2);
    smt_astt lo_r = ctx->mk_sub(iv1.lo, iv2.hi);
    smt_astt hi_r = ctx->mk_sub(iv1.hi, iv2.lo);
    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    store_interval(real_result, bounds.first, bounds.second);
    store_combined_nan_pred(real_result, side1, side2);
    return real_result;
  }
  return ctx->apply_ieee754_semantics(
    real_result, fbv_type, nullptr, rounding_mode);
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
    is_known_rounding_mode(rounding_mode) && is_single_or_double(fbv_type))
  {
    ra_interval_t iv1 = get_interval(side1);
    ra_interval_t iv2 = get_interval(side2);
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
    store_interval(real_result, bounds.first, bounds.second);
    store_combined_nan_pred(real_result, side1, side2);
    return real_result;
  }
  return ctx->apply_ieee754_semantics(
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
    return ctx->apply_ieee754_semantics(
      real_result, fbv_type, nullptr, to_ieee_div2t(expr).rounding_mode);
  }

  smt_astt sentinel = ctx->get_double_inf_sentinel();
  smt_astt inf_result =
    ctx->mk_ite(ctx->mk_lt(side1, zero), ctx->mk_sub(zero, sentinel), sentinel);
  smt_astt real_result = ctx->mk_div(side1, side2);
  const expr2tc &rounding_mode = to_ieee_div2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_known_rounding_mode(rounding_mode))
  {
    ra_interval_t iv1 = get_interval(side1);
    ra_interval_t iv2 = get_interval(side2);

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
    smt_astt a = ctx->mk_ite(div_by_zero, inf_result, real_result);
    store_interval(a, bounds.first, bounds.second);
    store_combined_nan_pred(a, side1, side2);
    return a;
  }

  smt_astt ieee_result =
    ctx->apply_ieee754_semantics(real_result, fbv_type, nullptr, rounding_mode);
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
  const expr2tc &rounding_mode = to_ieee_fma2t(expr).rounding_mode;

  if (
    ctx->options.get_bool_option("ir-ieee") &&
    is_known_rounding_mode(rounding_mode) && is_single_or_double(fbv_type))
  {
    ra_interval_t iv1 = get_interval(val1);
    ra_interval_t iv2 = get_interval(val2);
    ra_interval_t iv3 = get_interval(val3);

    // Multiplication hull: min/max over the four endpoint products of x and y.
    smt_astt p1 = ctx->mk_mul(iv1.lo, iv2.lo);
    smt_astt p2 = ctx->mk_mul(iv1.lo, iv2.hi);
    smt_astt p3 = ctx->mk_mul(iv1.hi, iv2.lo);
    smt_astt p4 = ctx->mk_mul(iv1.hi, iv2.hi);
    smt_astt mul_lo = ctx->mk_ite(
      ctx->mk_le(p1, p2),
      ctx->mk_ite(
        ctx->mk_le(p1, p3),
        ctx->mk_ite(ctx->mk_le(p1, p4), p1, p4),
        ctx->mk_ite(ctx->mk_le(p3, p4), p3, p4)),
      ctx->mk_ite(
        ctx->mk_le(p2, p3),
        ctx->mk_ite(ctx->mk_le(p2, p4), p2, p4),
        ctx->mk_ite(ctx->mk_le(p3, p4), p3, p4)));
    smt_astt mul_hi = ctx->mk_ite(
      ctx->mk_le(p2, p1),
      ctx->mk_ite(
        ctx->mk_le(p3, p1),
        ctx->mk_ite(ctx->mk_le(p4, p1), p1, p4),
        ctx->mk_ite(ctx->mk_le(p4, p3), p3, p4)),
      ctx->mk_ite(
        ctx->mk_le(p3, p2),
        ctx->mk_ite(ctx->mk_le(p4, p2), p2, p4),
        ctx->mk_ite(ctx->mk_le(p4, p3), p3, p4)));

    // Fused add: extend the multiplication hull by the z interval.
    // r = x*y + z is a single rounding, so the enclosure is applied once.
    smt_astt lo_r = ctx->mk_add(mul_lo, iv3.lo);
    smt_astt hi_r = ctx->mk_add(mul_hi, iv3.hi);

    auto bounds =
      apply_enclosure(real_result, lo_r, hi_r, fbv_type, rounding_mode);
    store_interval(real_result, bounds.first, bounds.second);
    smt_astt nan_p = combine_nan_preds(
      combine_nan_preds(get_nan_pred(val1), get_nan_pred(val2)),
      get_nan_pred(val3));
    if (nan_p)
      store_nan_pred(real_result, nan_p);
    return real_result;
  }

  return ctx->apply_ieee754_semantics(
    real_result, fbv_type, nullptr, rounding_mode);
}
