#include <solvers/smt/smt_conv.h>

static smt_astt extract_exponent(smt_convt *ctx, smt_astt fp)
{
  std::size_t exp_top = fp->sort->get_data_width() - 2;
  std::size_t exp_bot = fp->sort->get_significand_width() - 2;
  return ctx->mk_extract(fp, exp_top, exp_bot + 1);
}

static smt_astt extract_significand(smt_convt *ctx, smt_astt fp)
{
  return ctx->mk_extract(fp, fp->sort->get_significand_width() - 2, 0);
}

static smt_astt extract_signbit(smt_convt *ctx, smt_astt fp)
{
  return ctx->mk_extract(
    fp, fp->sort->get_data_width() - 1, fp->sort->get_data_width() - 1);
}

static smt_astt extract_exp_sig(smt_convt *ctx, smt_astt fp)
{
  return ctx->mk_extract(fp, fp->sort->get_data_width() - 2, 0);
}

fp_convt::fp_convt(smt_convt *_ctx) : ctx(_ctx)
{
}

smt_astt fp_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  smt_sortt s = ctx->mk_bvfp_sort(thereal.spec.e, thereal.spec.f);
  return ctx->mk_smt_bv(thereal.pack(), s);
}

smt_sortt fp_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  return ctx->mk_bvfp_sort(ew, sw);
}

smt_sortt fp_convt::mk_fpbv_rm_sort()
{
  return ctx->mk_bvfp_rm_sort();
}

smt_astt fp_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  // TODO: we always create the same positive NaN:
  // 01111111100000000000000000000001
  smt_astt top_exp = mk_top_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(0), 1),
      ctx->mk_concat(top_exp, ctx->mk_smt_bv(BigInt(1), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(sgn), 1),
      ctx->mk_concat(top_exp, ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  return ctx->mk_smt_bv(BigInt(rm), 3);
}

smt_astt fp_convt::mk_smt_nearbyint_from_float(smt_astt x, smt_astt rm)
{
  unsigned ebits = x->sort->get_exponent_width();
  unsigned sbits = x->sort->get_significand_width();

  smt_astt rm_is_rta = mk_is_rm(rm, ieee_floatt::ROUND_TO_AWAY);
  smt_astt rm_is_rte = mk_is_rm(rm, ieee_floatt::ROUND_TO_EVEN);
  smt_astt rm_is_rtp = mk_is_rm(rm, ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt rm_is_rtn = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt rm_is_rtz = mk_is_rm(rm, ieee_floatt::ROUND_TO_ZERO);

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);
  smt_astt nzero = mk_nzero(ebits, sbits);
  smt_astt pzero = mk_pzero(ebits, sbits);

  smt_astt x_is_neg = mk_is_neg(x);

  // (x is NaN) -> NaN
  smt_astt c1 = mk_smt_fpbv_is_nan(x);
  smt_astt v1 = nan;

  // (x is +-oo) -> x
  smt_astt c2 = mk_smt_fpbv_is_inf(x);
  smt_astt v2 = x;

  // (x is +-0) -> x ; -0.0 -> -0.0, says IEEE754, Sec 5.9.
  smt_astt c3 = mk_smt_fpbv_is_zero(x);
  smt_astt v3 = x;

  smt_astt one_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt zero_1 = ctx->mk_smt_bv(BigInt(0), 1);

  smt_astt a_sgn, a_sig, a_exp, a_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, true);

  smt_astt sgn_eq_1 = ctx->mk_eq(a_sgn, one_1);
  smt_astt xzero = ctx->mk_ite(sgn_eq_1, nzero, pzero);

  // exponent < 0 -> 0/1
  smt_astt exp_h = ctx->mk_extract(a_exp, ebits - 1, ebits - 1);
  smt_astt exp_lt_zero = ctx->mk_eq(exp_h, one_1);
  smt_astt c4 = exp_lt_zero;

  smt_astt pone = mk_one(zero_1, ebits, sbits);
  smt_astt none = mk_one(one_1, ebits, sbits);
  smt_astt xone = ctx->mk_ite(sgn_eq_1, none, pone);

  smt_astt pow_2_sbitsm1 =
    ctx->mk_smt_bv(BigInt(power2(sbits - 1, false)), sbits);
  smt_astt m1 = ctx->mk_smt_bv(BigInt(-1), ebits);
  smt_astt t1 = ctx->mk_eq(a_sig, pow_2_sbitsm1);
  smt_astt t2 = ctx->mk_eq(a_exp, m1);
  smt_astt tie = ctx->mk_and(t1, t2);

  smt_astt c421 = ctx->mk_and(tie, rm_is_rte);
  smt_astt c422 = ctx->mk_and(tie, rm_is_rta);
  smt_astt c423 = ctx->mk_bvsle(a_exp, ctx->mk_smt_bv(BigInt(-2), ebits));

  smt_astt v42 = xone;
  v42 = ctx->mk_ite(c423, xzero, v42);
  v42 = ctx->mk_ite(c422, xone, v42);
  v42 = ctx->mk_ite(c421, xzero, v42);

  smt_astt v4_rtp = ctx->mk_ite(x_is_neg, nzero, pone);
  smt_astt v4_rtn = ctx->mk_ite(x_is_neg, none, pzero);

  smt_astt v4 = ctx->mk_ite(rm_is_rtp, v4_rtp, v42);
  v4 = ctx->mk_ite(rm_is_rtn, v4_rtn, v4);
  v4 = ctx->mk_ite(rm_is_rtz, xzero, v4);

  // exponent >= sbits-1
  smt_astt exp_is_large =
    ctx->mk_bvsle(ctx->mk_smt_bv(BigInt(sbits - 1), ebits), a_exp);
  smt_astt c5 = exp_is_large;
  smt_astt v5 = x;

  // Actual conversion with rounding.
  // x.exponent >= 0 && x.exponent < x.sbits - 1
  smt_astt res_sgn = a_sgn;
  smt_astt res_exp = a_exp;

  assert(a_sig->sort->get_data_width() == sbits);
  assert(a_exp->sort->get_data_width() == ebits);

  smt_astt zero_s = ctx->mk_smt_bv(BigInt(0), sbits);

  smt_astt shift = ctx->mk_bvsub(
    ctx->mk_smt_bv(BigInt(sbits - 1), sbits),
    ctx->mk_zero_ext(a_exp, sbits - ebits));
  smt_astt shifted_sig = ctx->mk_bvlshr(
    ctx->mk_concat(a_sig, zero_s), ctx->mk_concat(zero_s, shift));
  smt_astt div = ctx->mk_extract(shifted_sig, 2 * sbits - 1, sbits);
  smt_astt rem = ctx->mk_extract(shifted_sig, sbits - 1, 0);

  assert(shift->sort->get_data_width() == sbits);
  assert(div->sort->get_data_width() == sbits);
  assert(rem->sort->get_data_width() == sbits);

  smt_astt div_p1 = ctx->mk_bvadd(div, ctx->mk_smt_bv(BigInt(1), sbits));

  smt_astt tie_pttrn =
    ctx->mk_concat(one_1, ctx->mk_smt_bv(BigInt(0), sbits - 1));
  smt_astt tie2 = ctx->mk_eq(rem, tie_pttrn);
  smt_astt div_last = ctx->mk_extract(div, 0, 0);
  smt_astt div_last_eq_1 = ctx->mk_eq(div_last, one_1);
  smt_astt rte_and_dl_eq_1 = ctx->mk_and(rm_is_rte, div_last_eq_1);
  smt_astt rte_and_dl_eq_1_or_rta = ctx->mk_or(rte_and_dl_eq_1, rm_is_rta);
  smt_astt tie_pttrn_ule_rem = ctx->mk_bvule(tie_pttrn, rem);
  smt_astt tie2_c =
    ctx->mk_ite(tie2, rte_and_dl_eq_1_or_rta, tie_pttrn_ule_rem);
  smt_astt v51 = ctx->mk_ite(tie2_c, div_p1, div);

  smt_astt rem_eq_0 = ctx->mk_eq(rem, ctx->mk_smt_bv(BigInt(0), sbits));
  smt_astt sgn_eq_zero = ctx->mk_eq(res_sgn, zero_1);
  smt_astt c521 = ctx->mk_not(rem_eq_0);
  c521 = ctx->mk_and(c521, sgn_eq_zero);
  smt_astt v52 = ctx->mk_ite(c521, div_p1, div);

  smt_astt sgn_eq_one = ctx->mk_eq(res_sgn, one_1);
  smt_astt c531 = ctx->mk_not(rem_eq_0);
  c531 = ctx->mk_and(c531, sgn_eq_one);
  smt_astt v53 = ctx->mk_ite(c531, div_p1, div);

  smt_astt c51 = ctx->mk_or(rm_is_rte, rm_is_rta);
  smt_astt c52 = rm_is_rtp;
  smt_astt c53 = rm_is_rtn;

  smt_astt res_sig = div;
  res_sig = ctx->mk_ite(c53, v53, res_sig);
  res_sig = ctx->mk_ite(c52, v52, res_sig);
  res_sig = ctx->mk_ite(c51, v51, res_sig);
  res_sig = ctx->mk_zero_ext(
    ctx->mk_concat(res_sig, ctx->mk_smt_bv(BigInt(0), 3)),
    1); // rounding bits are all 0.

  assert(res_exp->sort->get_data_width() == ebits);
  assert(shift->sort->get_data_width() == sbits);

  smt_astt e_shift = (ebits + 2 <= sbits + 1)
                       ? ctx->mk_extract(shift, ebits + 1, 0)
                       : ctx->mk_sign_ext(shift, (ebits + 2) - (sbits));
  assert(e_shift->sort->get_data_width() == ebits + 2);
  res_exp = ctx->mk_bvadd(ctx->mk_zero_ext(res_exp, 2), e_shift);

  assert(res_sgn->sort->get_data_width() == 1);
  assert(res_sig->sort->get_data_width() == sbits + 4);
  assert(res_exp->sort->get_data_width() == ebits + 2);

  // CMW: We use the rounder for normalization.
  smt_astt v6;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, v6);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c5, v5, v6);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt fp_convt::mk_smt_fpbv_sqrt(smt_astt x, smt_astt rm)
{
  unsigned ebits = x->sort->get_exponent_width();
  unsigned sbits = x->sort->get_significand_width();

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);

  smt_astt zero1 = ctx->mk_smt_bv(BigInt(0), 1);
  smt_astt one1 = ctx->mk_smt_bv(BigInt(1), 1);

  // (x is NaN) -> NaN
  smt_astt c1 = x_is_nan;
  smt_astt v1 = x;

  // (x is +oo) -> +oo
  smt_astt c2 = mk_is_pinf(x);
  smt_astt v2 = x;

  // (x is +-0) -> +-0
  smt_astt c3 = mk_smt_fpbv_is_zero(x);
  smt_astt v3 = x;

  // (x < 0) -> NaN
  smt_astt c4 = mk_is_neg(x);
  smt_astt v4 = nan;

  // else comes the actual square root.

  smt_astt a_sgn, a_sig, a_exp, a_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, true);

  assert(a_sig->sort->get_data_width() == sbits);
  assert(a_exp->sort->get_data_width() == ebits);

  smt_astt res_sgn = zero1;

  smt_astt real_exp =
    ctx->mk_bvsub(ctx->mk_sign_ext(a_exp, 1), ctx->mk_zero_ext(a_lz, 1));
  smt_astt res_exp = ctx->mk_sign_ext(ctx->mk_extract(real_exp, ebits, 1), 2);

  smt_astt e_is_odd = ctx->mk_eq(ctx->mk_extract(real_exp, 0, 0), one1);

  smt_astt a_z = ctx->mk_concat(a_sig, zero1);
  smt_astt z_a = ctx->mk_concat(zero1, a_sig);
  smt_astt sig_prime = ctx->mk_ite(e_is_odd, a_z, z_a);
  assert(sig_prime->sort->get_data_width() == sbits + 1);

  // This is algorithm 10.2 in the Handbook of Floating-Point Arithmetic
  auto p2 = power2(sbits + 3, false);
  smt_astt Q = ctx->mk_smt_bv(BigInt(p2), sbits + 5);
  smt_astt R =
    ctx->mk_bvsub(ctx->mk_concat(sig_prime, ctx->mk_smt_bv(BigInt(0), 4)), Q);
  smt_astt S = Q;

  smt_astt T;
  for(unsigned i = 0; i < sbits + 3; i++)
  {
    S = ctx->mk_concat(zero1, ctx->mk_extract(S, sbits + 4, 1));

    smt_astt twoQ_plus_S =
      ctx->mk_bvadd(ctx->mk_concat(Q, zero1), ctx->mk_concat(zero1, S));
    T = ctx->mk_bvsub(ctx->mk_concat(R, zero1), twoQ_plus_S);

    assert(Q->sort->get_data_width() == sbits + 5);
    assert(R->sort->get_data_width() == sbits + 5);
    assert(S->sort->get_data_width() == sbits + 5);
    assert(T->sort->get_data_width() == sbits + 6);

    smt_astt T_lsds5 = ctx->mk_extract(T, sbits + 5, sbits + 5);
    smt_astt t_lt_0 = ctx->mk_eq(T_lsds5, one1);

    smt_astt Q_or_S = ctx->mk_bvor(Q, S);
    Q = ctx->mk_ite(t_lt_0, Q, Q_or_S);
    smt_astt R_shftd = ctx->mk_concat(ctx->mk_extract(R, sbits + 3, 0), zero1);
    smt_astt T_lsds4 = ctx->mk_extract(T, sbits + 4, 0);
    R = ctx->mk_ite(t_lt_0, R_shftd, T_lsds4);
  }

  smt_astt zero_sbits5 = ctx->mk_smt_bv(BigInt(0), sbits + 5);
  smt_astt is_exact = ctx->mk_eq(R, zero_sbits5);

  smt_astt last = ctx->mk_extract(Q, 0, 0);
  smt_astt rest = ctx->mk_extract(Q, sbits + 3, 1);
  smt_astt rest_ext = ctx->mk_zero_ext(rest, 1);
  smt_astt last_ext = ctx->mk_zero_ext(last, sbits + 3);
  smt_astt one_sbits4 = ctx->mk_smt_bv(BigInt(1), sbits + 4);
  smt_astt sticky = ctx->mk_ite(is_exact, last_ext, one_sbits4);
  smt_astt res_sig = ctx->mk_bvor(rest_ext, sticky);

  assert(res_sig->sort->get_data_width() == sbits + 4);

  smt_astt rounded;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, rounded);
  smt_astt v5 = rounded;

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c4, v4, v5);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt
fp_convt::mk_smt_fpbv_fma(smt_astt x, smt_astt y, smt_astt z, smt_astt rm)
{
  assert(x->sort->get_data_width() == y->sort->get_data_width());
  assert(x->sort->get_exponent_width() == y->sort->get_exponent_width());
  assert(x->sort->get_data_width() == z->sort->get_data_width());
  assert(x->sort->get_exponent_width() == z->sort->get_exponent_width());

  unsigned ebits = x->sort->get_exponent_width();
  unsigned sbits = x->sort->get_significand_width();

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);
  smt_astt nzero = mk_nzero(ebits, sbits);
  smt_astt pzero = mk_pzero(ebits, sbits);
  smt_astt ninf = mk_ninf(ebits, sbits);
  smt_astt pinf = mk_pinf(ebits, sbits);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt x_is_pos = mk_is_pos(x);
  smt_astt x_is_neg = mk_is_neg(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt y_is_pos = mk_is_pos(y);
  smt_astt y_is_neg = mk_is_neg(y);
  smt_astt z_is_nan = mk_smt_fpbv_is_nan(z);
  smt_astt z_is_zero = mk_smt_fpbv_is_zero(z);
  smt_astt z_is_neg = mk_is_neg(z);
  smt_astt z_is_inf = mk_smt_fpbv_is_inf(z);

  smt_astt rm_is_to_neg = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);

  smt_astt inf_xor = ctx->mk_xor(x_is_neg, y_is_neg);
  inf_xor = ctx->mk_xor(inf_xor, z_is_neg);
  smt_astt inf_cond = ctx->mk_and(z_is_inf, inf_xor);

  // (x is NaN) || (y is NaN) || (z is Nan) -> NaN
  smt_astt c1 = ctx->mk_or(ctx->mk_or(x_is_nan, y_is_nan), z_is_nan);
  smt_astt v1 = nan;

  // (x is +oo) -> if (y is 0) then NaN else inf with y's sign.
  smt_astt c2 = mk_is_pinf(x);
  smt_astt y_sgn_inf = ctx->mk_ite(y_is_pos, pinf, ninf);
  smt_astt inf_or = ctx->mk_or(y_is_zero, inf_cond);
  smt_astt v2 = ctx->mk_ite(inf_or, nan, y_sgn_inf);

  // (y is +oo) -> if (x is 0) then NaN else inf with x's sign.
  smt_astt c3 = mk_is_pinf(y);
  smt_astt x_sgn_inf = ctx->mk_ite(x_is_pos, pinf, ninf);
  inf_or = ctx->mk_or(x_is_zero, inf_cond);
  smt_astt v3 = ctx->mk_ite(inf_or, nan, x_sgn_inf);

  // (x is -oo) -> if (y is 0) then NaN else inf with -y's sign.
  smt_astt c4 = mk_is_ninf(x);
  smt_astt neg_y_sgn_inf = ctx->mk_ite(y_is_pos, ninf, pinf);
  inf_or = ctx->mk_or(y_is_zero, inf_cond);
  smt_astt v4 = ctx->mk_ite(inf_or, nan, neg_y_sgn_inf);

  // (y is -oo) -> if (x is 0) then NaN else inf with -x's sign.
  smt_astt c5 = mk_is_ninf(y);
  smt_astt neg_x_sgn_inf = ctx->mk_ite(x_is_pos, ninf, pinf);
  inf_or = ctx->mk_or(x_is_zero, inf_cond);
  smt_astt v5 = ctx->mk_ite(inf_or, nan, neg_x_sgn_inf);

  // z is +-INF -> z.
  smt_astt c6 = mk_smt_fpbv_is_inf(z);
  smt_astt v6 = z;

  // (x is 0) || (y is 0) -> z
  smt_astt c7 = ctx->mk_or(x_is_zero, y_is_zero);
  smt_astt xy_sgn = ctx->mk_xor(x_is_neg, y_is_neg);

  smt_astt xyz_sgn = ctx->mk_xor(xy_sgn, z_is_neg);
  smt_astt c71 = ctx->mk_and(z_is_zero, xyz_sgn);

  smt_astt zero_cond = ctx->mk_ite(rm_is_to_neg, nzero, pzero);
  smt_astt v7 = ctx->mk_ite(c71, zero_cond, z);

  // else comes the fused multiplication.
  smt_astt one_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt zero_1 = ctx->mk_smt_bv(BigInt(0), 1);

  smt_astt a_sgn, a_sig, a_exp, a_lz;
  smt_astt b_sgn, b_sig, b_exp, b_lz;
  smt_astt c_sgn, c_sig, c_exp, c_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, true);
  unpack(y, b_sgn, b_sig, b_exp, b_lz, true);
  unpack(z, c_sgn, c_sig, c_exp, c_lz, true);

  smt_astt a_lz_ext = ctx->mk_zero_ext(a_lz, 2);
  smt_astt b_lz_ext = ctx->mk_zero_ext(b_lz, 2);
  smt_astt c_lz_ext = ctx->mk_zero_ext(c_lz, 2);

  smt_astt a_sig_ext = ctx->mk_zero_ext(a_sig, sbits);
  smt_astt b_sig_ext = ctx->mk_zero_ext(b_sig, sbits);

  smt_astt a_exp_ext = ctx->mk_sign_ext(a_exp, 2);
  smt_astt b_exp_ext = ctx->mk_sign_ext(b_exp, 2);
  smt_astt c_exp_ext = ctx->mk_sign_ext(c_exp, 2);

  smt_astt mul_sgn = ctx->mk_bvxor(a_sgn, b_sgn);

  smt_astt mul_exp = ctx->mk_bvadd(
    ctx->mk_bvsub(a_exp_ext, a_lz_ext), ctx->mk_bvsub(b_exp_ext, b_lz_ext));

  smt_astt mul_sig = ctx->mk_bvmul(a_sig_ext, b_sig_ext);

  assert(mul_sig->sort->get_data_width() == 2 * sbits);
  assert(mul_exp->sort->get_data_width() == ebits + 2);

  // The product has the form [-1][0].[2*sbits - 2].

  // Extend c
  smt_astt c_sig_ext = ctx->mk_zero_ext(
    ctx->mk_concat(c_sig, ctx->mk_smt_bv(BigInt(0), sbits + 2)), 1);
  c_exp_ext = ctx->mk_bvsub(c_exp_ext, c_lz_ext);
  mul_sig = ctx->mk_concat(mul_sig, ctx->mk_smt_bv(BigInt(0), 3));

  assert(mul_sig->sort->get_data_width() == 2 * sbits + 3);
  assert(c_sig_ext->sort->get_data_width() == 2 * sbits + 3);

  smt_astt swap_cond = ctx->mk_bvsle(mul_exp, c_exp_ext);

  smt_astt e_sgn = ctx->mk_ite(swap_cond, c_sgn, mul_sgn);
  smt_astt e_sig =
    ctx->mk_ite(swap_cond, c_sig_ext, mul_sig); // has 2 * sbits + 3
  smt_astt e_exp = ctx->mk_ite(swap_cond, c_exp_ext, mul_exp); // has ebits + 2
  smt_astt f_sgn = ctx->mk_ite(swap_cond, mul_sgn, c_sgn);
  smt_astt f_sig =
    ctx->mk_ite(swap_cond, mul_sig, c_sig_ext); // has 2 * sbits + 3
  smt_astt f_exp = ctx->mk_ite(swap_cond, mul_exp, c_exp_ext); // has ebits + 2

  assert(e_sig->sort->get_data_width() == 2 * sbits + 3);
  assert(f_sig->sort->get_data_width() == 2 * sbits + 3);
  assert(e_exp->sort->get_data_width() == ebits + 2);
  assert(f_exp->sort->get_data_width() == ebits + 2);

  smt_astt exp_delta = ctx->mk_bvsub(e_exp, f_exp);

  // cap the delta

  smt_astt cap = ctx->mk_smt_bv(BigInt(2 * sbits + 3), ebits + 2);
  smt_astt cap_le_delta = ctx->mk_bvule(cap, exp_delta);
  exp_delta = ctx->mk_ite(cap_le_delta, cap, exp_delta);
  assert(exp_delta->sort->get_data_width() == ebits + 2);

  // Alignment shift with sticky bit computation.
  smt_astt shifted_big = ctx->mk_bvlshr(
    ctx->mk_concat(f_sig, ctx->mk_smt_bv(BigInt(0), sbits)),
    ctx->mk_zero_ext(exp_delta, (3 * sbits + 3) - (ebits + 2)));
  smt_astt shifted_f_sig = ctx->mk_extract(shifted_big, 3 * sbits + 2, sbits);
  smt_astt alignment_sticky_raw = ctx->mk_extract(shifted_big, sbits - 1, 0);
  smt_astt alignment_sticky = ctx->mk_bvredor(alignment_sticky_raw);
  assert(shifted_f_sig->sort->get_data_width() == 2 * sbits + 3);

  // Significant addition.
  // Two extra bits for the sign and for catching overflows.
  e_sig = ctx->mk_zero_ext(e_sig, 2);
  shifted_f_sig = ctx->mk_zero_ext(shifted_f_sig, 2);

  smt_astt eq_sgn = ctx->mk_eq(e_sgn, f_sgn);

  assert(e_sig->sort->get_data_width() == 2 * sbits + 5);
  assert(shifted_f_sig->sort->get_data_width() == 2 * sbits + 5);

  smt_astt sticky_wide = ctx->mk_zero_ext(alignment_sticky, 2 * sbits + 4);
  smt_astt e_plus_f = ctx->mk_bvadd(e_sig, shifted_f_sig);
  e_plus_f = ctx->mk_ite(
    ctx->mk_eq(ctx->mk_extract(e_plus_f, 0, 0), zero_1),
    ctx->mk_bvadd(e_plus_f, sticky_wide),
    e_plus_f);
  smt_astt e_minus_f = ctx->mk_bvsub(e_sig, shifted_f_sig);
  e_minus_f = ctx->mk_ite(
    ctx->mk_eq(ctx->mk_extract(e_minus_f, 0, 0), zero_1),
    ctx->mk_bvsub(e_minus_f, sticky_wide),
    e_minus_f);

  smt_astt sum = ctx->mk_ite(eq_sgn, e_plus_f, e_minus_f);
  assert(sum->sort->get_data_width() == 2 * sbits + 5);

  smt_astt sign_bv = ctx->mk_extract(sum, 2 * sbits + 4, 2 * sbits + 4);
  smt_astt n_sum = ctx->mk_bvneg(sum);

  smt_astt res_sig_eq = ctx->mk_eq(sign_bv, one_1);
  smt_astt sig_abs = ctx->mk_ite(res_sig_eq, n_sum, sum);

  smt_astt not_e_sgn = ctx->mk_bvnot(e_sgn);
  smt_astt not_f_sgn = ctx->mk_bvnot(f_sgn);
  smt_astt not_sign_bv = ctx->mk_bvnot(sign_bv);

  smt_astt res_sgn_c1 = ctx->mk_bvand(ctx->mk_bvand(not_e_sgn, f_sgn), sign_bv);
  smt_astt res_sgn_c2 =
    ctx->mk_bvand(ctx->mk_bvand(e_sgn, not_f_sgn), not_sign_bv);
  smt_astt res_sgn_c3 = ctx->mk_bvand(e_sgn, f_sgn);
  smt_astt res_sgn =
    ctx->mk_bvor(ctx->mk_bvor(res_sgn_c1, res_sgn_c2), res_sgn_c3);

  smt_astt is_sig_neg =
    ctx->mk_eq(one_1, ctx->mk_extract(sig_abs, 2 * sbits + 4, 2 * sbits + 4));
  sig_abs = ctx->mk_ite(is_sig_neg, ctx->mk_bvneg(sig_abs), sig_abs);

  // Result could have overflown into 4.xxx.
  assert(sig_abs->sort->get_data_width() == 2 * sbits + 5);
  smt_astt extra = ctx->mk_extract(sig_abs, 2 * sbits + 4, 2 * sbits + 3);
  smt_astt extra_is_zero = ctx->mk_eq(extra, ctx->mk_smt_bv(BigInt(0), 2));

  smt_astt res_exp = ctx->mk_ite(
    extra_is_zero,
    e_exp,
    ctx->mk_bvadd(e_exp, ctx->mk_smt_bv(BigInt(1), ebits + 2)));

  // Renormalize
  smt_astt zero_e2 = ctx->mk_smt_bv(BigInt(0), ebits + 2);
  smt_astt min_exp = mk_min_exp(ebits);
  min_exp = ctx->mk_sign_ext(min_exp, 2);
  smt_astt sig_lz = mk_leading_zeros(sig_abs, ebits + 2);
  sig_lz = ctx->mk_bvsub(sig_lz, ctx->mk_smt_bv(BigInt(2), ebits + 2));
  smt_astt max_exp_delta = ctx->mk_bvsub(res_exp, min_exp);
  smt_astt sig_lz_capped =
    ctx->mk_ite(ctx->mk_bvsle(sig_lz, max_exp_delta), sig_lz, max_exp_delta);
  smt_astt renorm_delta =
    ctx->mk_ite(ctx->mk_bvsle(zero_e2, sig_lz_capped), sig_lz_capped, zero_e2);
  res_exp = ctx->mk_bvsub(res_exp, renorm_delta);
  sig_abs = ctx->mk_bvshl(
    sig_abs, ctx->mk_zero_ext(renorm_delta, 2 * sbits + 3 - ebits));

  unsigned too_short = 0;
  if(sbits < 5)
  {
    too_short = 6 - sbits + 1;
    sig_abs = ctx->mk_concat(sig_abs, ctx->mk_smt_bv(BigInt(0), too_short));
  }

  smt_astt sticky_h1 = ctx->mk_extract(sig_abs, sbits + too_short - 2, 0);
  smt_astt sig_abs_h1 =
    ctx->mk_extract(sig_abs, 2 * sbits + too_short + 4, sbits - 1 + too_short);
  smt_astt sticky_h1_red =
    ctx->mk_zero_ext(ctx->mk_bvredor(sticky_h1), sbits + 5);
  smt_astt sig_abs_h1_f = ctx->mk_bvor(sig_abs_h1, sticky_h1_red);
  smt_astt res_sig_1 = ctx->mk_extract(sig_abs_h1_f, sbits + 3, 0);
  assert(sticky_h1->sort->get_data_width() == sbits + too_short - 1);
  assert(sig_abs_h1->sort->get_data_width() == sbits + 6);
  assert(sticky_h1_red->sort->get_data_width() == sbits + 6);
  assert(sig_abs_h1_f->sort->get_data_width() == sbits + 6);
  assert(res_sig_1->sort->get_data_width() == sbits + 4);

  smt_astt sig_abs_h2 =
    ctx->mk_extract(sig_abs, 2 * sbits + too_short + 4, sbits + too_short);
  smt_astt sticky_h2_red =
    ctx->mk_zero_ext(ctx->mk_bvredor(sticky_h1), sbits + 4);
  smt_astt sig_abs_h2_f =
    ctx->mk_zero_ext(ctx->mk_bvor(sig_abs_h2, sticky_h2_red), 1);
  smt_astt res_sig_2 = ctx->mk_extract(sig_abs_h2_f, sbits + 3, 0);
  assert(sig_abs_h2->sort->get_data_width() == sbits + 5);
  assert(sticky_h2_red->sort->get_data_width() == sbits + 5);
  assert(sig_abs_h2_f->sort->get_data_width() == sbits + 6);
  assert(res_sig_2->sort->get_data_width() == sbits + 4);

  smt_astt res_sig = ctx->mk_ite(extra_is_zero, res_sig_1, res_sig_2);

  assert(res_sig->sort->get_data_width() == sbits + 4);

  smt_astt nil_sbits4 = ctx->mk_smt_bv(BigInt(0), sbits + 4);
  smt_astt is_zero_sig = ctx->mk_eq(res_sig, nil_sbits4);

  smt_astt zero_case = ctx->mk_ite(rm_is_to_neg, nzero, pzero);

  smt_astt rounded;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, rounded);

  smt_astt v8 = ctx->mk_ite(is_zero_sig, zero_case, rounded);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c7, v7, v8);
  result = ctx->mk_ite(c6, v6, result);
  result = ctx->mk_ite(c5, v5, result);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt fp_convt::mk_to_bv(smt_astt x, bool is_signed, std::size_t width)
{
  smt_astt rm = mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  smt_sortt xs = x->sort;

  unsigned ebits = xs->get_exponent_width();
  unsigned sbits = xs->get_significand_width();
  unsigned bv_sz = width;

  smt_astt bv0 = ctx->mk_smt_bv(BigInt(0), 1);
  smt_astt bv1 = ctx->mk_smt_bv(BigInt(1), 1);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt x_is_inf = mk_smt_fpbv_is_inf(x);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt x_is_neg = mk_is_neg(x);

  // NaN, Inf, or negative (except -0) -> unspecified
  smt_astt c1 = ctx->mk_or(x_is_nan, x_is_inf);
  smt_astt unspec_v = ctx->mk_smt_symbol("UNSPEC_FP", ctx->mk_bv_sort(width));
  smt_astt v1 = unspec_v;

  // +-0 -> 0
  smt_astt c2 = x_is_zero;
  smt_astt v2 = ctx->mk_smt_bv(BigInt(0), width);

  // Otherwise...
  smt_astt sgn, sig, exp, lz;
  unpack(x, sgn, sig, exp, lz, true);

  // sig is of the form +- [1].[sig] * 2^(exp-lz)
  assert(sgn->sort->get_data_width() == 1);
  assert(sig->sort->get_data_width() == sbits);
  assert(exp->sort->get_data_width() == ebits);
  assert(lz->sort->get_data_width() == ebits);

  unsigned sig_sz = sbits;
  if(sig_sz < (bv_sz + 3))
    sig = ctx->mk_concat(sig, ctx->mk_smt_bv(BigInt(0), bv_sz - sig_sz + 3));
  sig_sz = sig->sort->get_data_width();
  assert(sig_sz >= (bv_sz + 3));

  // x is of the form +- [1].[sig][r][g][s] ... and at least bv_sz + 3 long
  smt_astt exp_m_lz =
    ctx->mk_bvsub(ctx->mk_sign_ext(exp, 2), ctx->mk_zero_ext(lz, 2));

  // big_sig is +- [... bv_sz+2 bits ...][1].[r][ ... sbits-1  ... ]
  smt_astt big_sig = ctx->mk_concat(ctx->mk_zero_ext(sig, bv_sz + 2), bv0);
  unsigned big_sig_sz = sig_sz + 1 + bv_sz + 2;
  assert(big_sig->sort->get_data_width() == big_sig_sz);

  smt_astt is_neg_shift =
    ctx->mk_bvsle(exp_m_lz, ctx->mk_smt_bv(BigInt(0), ebits + 2));
  smt_astt shift = ctx->mk_ite(is_neg_shift, ctx->mk_bvneg(exp_m_lz), exp_m_lz);
  if(ebits + 2 < big_sig_sz)
    shift = ctx->mk_zero_ext(shift, big_sig_sz - ebits - 2);
  else if(ebits + 2 > big_sig_sz)
  {
    smt_astt upper = ctx->mk_extract(shift, big_sig_sz, ebits + 2);
    shift = ctx->mk_extract(shift, ebits + 1, 0);
    shift = ctx->mk_ite(
      ctx->mk_eq(
        upper, ctx->mk_smt_bv(BigInt(0), upper->sort->get_data_width())),
      shift,
      ctx->mk_smt_bv(BigInt(big_sig_sz - 1), ebits + 2));
  }
  assert(shift->sort->get_data_width() == big_sig->sort->get_data_width());

  smt_astt shift_limit =
    ctx->mk_smt_bv(BigInt(bv_sz + 2), shift->sort->get_data_width());
  shift = ctx->mk_ite(ctx->mk_bvule(shift, shift_limit), shift, shift_limit);

  smt_astt big_sig_shifted = ctx->mk_ite(
    is_neg_shift,
    ctx->mk_bvlshr(big_sig, shift),
    ctx->mk_bvshl(big_sig, shift));
  smt_astt int_part =
    ctx->mk_extract(big_sig_shifted, big_sig_sz - 1, big_sig_sz - (bv_sz + 3));
  assert(int_part->sort->get_data_width() == bv_sz + 3);
  smt_astt last = ctx->mk_extract(
    big_sig_shifted, big_sig_sz - (bv_sz + 3), big_sig_sz - (bv_sz + 3));
  smt_astt round = ctx->mk_extract(
    big_sig_shifted, big_sig_sz - (bv_sz + 4), big_sig_sz - (bv_sz + 4));
  smt_astt stickies =
    ctx->mk_extract(big_sig_shifted, big_sig_sz - (bv_sz + 5), 0);
  smt_astt sticky = ctx->mk_bvredor(stickies);

  smt_astt rounding_decision =
    mk_rounding_decision(rm, sgn, last, round, sticky);
  assert(rounding_decision->sort->get_data_width() == 1);

  smt_astt inc = ctx->mk_zero_ext(rounding_decision, bv_sz + 2);
  smt_astt pre_rounded = ctx->mk_bvadd(int_part, inc);

  pre_rounded = ctx->mk_ite(x_is_neg, ctx->mk_bvneg(pre_rounded), pre_rounded);

  smt_astt ll, ul;
  if(!is_signed)
  {
    ll = ctx->mk_smt_bv(BigInt(0), bv_sz + 3);
    ul = ctx->mk_zero_ext(ctx->mk_smt_bv(BigInt(ULLONG_MAX), bv_sz), 3);
  }
  else
  {
    ll = ctx->mk_sign_ext(
      ctx->mk_concat(bv1, ctx->mk_smt_bv(BigInt(0), bv_sz - 1)), 3);
    ul = ctx->mk_zero_ext(ctx->mk_smt_bv(BigInt(ULLONG_MAX), bv_sz - 1), 4);
  }
  smt_astt in_range =
    ctx->mk_and(ctx->mk_bvsle(ll, pre_rounded), ctx->mk_bvsle(pre_rounded, ul));

  smt_astt rounded = ctx->mk_extract(pre_rounded, bv_sz - 1, 0);

  smt_astt result = ctx->mk_ite(ctx->mk_not(in_range), unspec_v, rounded);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt
fp_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width)
{
  return mk_to_bv(from, false, width);
}

smt_astt
fp_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width)
{
  return mk_to_bv(from, true, width);
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt x,
  smt_sortt to,
  smt_astt rm)
{
  unsigned from_sbits = x->sort->get_significand_width();
  unsigned from_ebits = x->sort->get_exponent_width();
  unsigned to_sbits = to->get_significand_width();
  unsigned to_ebits = to->get_exponent_width();

  if(from_sbits == to_sbits && from_ebits == to_ebits)
    return x;

  smt_astt one1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt pinf = mk_pinf(to_ebits, to_sbits);
  smt_astt ninf = mk_ninf(to_ebits, to_sbits);

  // NaN -> NaN
  smt_astt c1 = mk_smt_fpbv_is_nan(x);
  smt_astt v1 = mk_smt_fpbv_nan(to_ebits, to_sbits);

  // +0 -> +0
  smt_astt c2 = mk_is_pzero(x);
  smt_astt v2 = mk_pzero(to_ebits, to_sbits);

  // -0 -> -0
  smt_astt c3 = mk_is_nzero(x);
  smt_astt v3 = mk_nzero(to_ebits, to_sbits);

  // +oo -> +oo
  smt_astt c4 = mk_is_pinf(x);
  smt_astt v4 = pinf;

  // -oo -> -oo
  smt_astt c5 = mk_is_ninf(x);
  smt_astt v5 = ninf;

  // otherwise: the actual conversion with rounding.
  smt_astt sgn, sig, exp, lz;
  unpack(x, sgn, sig, exp, lz, true);

  smt_astt res_sgn = sgn;

  assert(sgn->sort->get_data_width() == 1);
  assert(sig->sort->get_data_width() == from_sbits);
  assert(exp->sort->get_data_width() == from_ebits);
  assert(lz->sort->get_data_width() == from_ebits);

  smt_astt res_sig;
  if(from_sbits < (to_sbits + 3))
  {
    // make sure that sig has at least to_sbits + 3
    res_sig =
      ctx->mk_concat(sig, ctx->mk_smt_bv(BigInt(0), to_sbits + 3 - from_sbits));
  }
  else if(from_sbits > (to_sbits + 3))
  {
    // collapse the extra bits into a sticky bit.
    smt_astt high =
      ctx->mk_extract(sig, from_sbits - 1, from_sbits - to_sbits - 2);
    assert(high->sort->get_data_width() == to_sbits + 2);
    smt_astt low = ctx->mk_extract(sig, from_sbits - to_sbits - 3, 0);
    smt_astt sticky = ctx->mk_bvredor(low);
    assert(sticky->sort->get_data_width() == 1);
    res_sig = ctx->mk_concat(high, sticky);
    assert(res_sig->sort->get_data_width() == to_sbits + 3);
  }
  else
    res_sig = sig;

  // extra zero in the front for the rounder.
  res_sig = ctx->mk_zero_ext(res_sig, 1);
  assert(res_sig->sort->get_data_width() == to_sbits + 4);

  smt_astt exponent_overflow = ctx->mk_smt_bool(false);

  smt_astt res_exp;
  if(from_ebits < (to_ebits + 2))
  {
    res_exp = ctx->mk_sign_ext(exp, to_ebits - from_ebits + 2);

    // subtract lz for subnormal numbers.
    smt_astt lz_ext = ctx->mk_zero_ext(lz, to_ebits - from_ebits + 2);
    res_exp = ctx->mk_bvsub(res_exp, lz_ext);
  }
  else if(from_ebits > (to_ebits + 2))
  {
    unsigned ebits_diff = from_ebits - (to_ebits + 2);

    // subtract lz for subnormal numbers.
    smt_astt exp_sub_lz =
      ctx->mk_bvsub(ctx->mk_sign_ext(exp, 2), ctx->mk_sign_ext(lz, 2));

    // check whether exponent is within roundable (to_ebits+2) range.
    BigInt z = power2(to_ebits + 1, true);
    smt_astt max_exp = ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(power2m1(to_ebits, false)), to_ebits + 1),
      ctx->mk_smt_bv(BigInt(0), 1));
    smt_astt min_exp = ctx->mk_smt_bv(BigInt(z + 2), to_ebits + 2);

    BigInt ovft = power2m1(to_ebits + 1, false);
    smt_astt first_ovf_exp = ctx->mk_smt_bv(BigInt(ovft), from_ebits + 2);
    smt_astt first_udf_exp = ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(-1), ebits_diff + 3),
      ctx->mk_smt_bv(BigInt(1), to_ebits + 1));

    smt_astt exp_in_range = ctx->mk_extract(exp_sub_lz, to_ebits + 1, 0);
    assert(exp_in_range->sort->get_data_width() == to_ebits + 2);

    smt_astt ovf_cond = ctx->mk_bvsle(first_ovf_exp, exp_sub_lz);
    smt_astt udf_cond = ctx->mk_bvsle(exp_sub_lz, first_udf_exp);

    res_exp = exp_in_range;
    res_exp = ctx->mk_ite(ovf_cond, max_exp, res_exp);
    res_exp = ctx->mk_ite(udf_cond, min_exp, res_exp);
  }
  else
  {
    // from_ebits == (to_ebits + 2)
    res_exp = ctx->mk_bvsub(exp, lz);
  }

  assert(res_exp->sort->get_data_width() == to_ebits + 2);

  smt_astt rounded;
  round(rm, res_sgn, res_sig, res_exp, to_ebits, to_sbits, rounded);

  smt_astt is_neg = ctx->mk_eq(sgn, one1);
  smt_astt sig_inf = ctx->mk_ite(is_neg, ninf, pinf);

  smt_astt v6 = ctx->mk_ite(exponent_overflow, sig_inf, rounded);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c5, v5, v6);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt
fp_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt x, smt_sortt to, smt_astt rm)
{
  // This is a conversion from unsigned bitvector to float:
  // ((_ to_fp_unsigned eb sb) RoundingMode (_ BitVec m) (_ FloatingPoint eb sb))
  // Semantics:
  //    Let b in[[(_ BitVec m)]] and let n be the unsigned integer represented by b.
  //    [[(_ to_fp_unsigned eb sb)]](r, x) = +infinity if n is too large to be
  //    represented as a finite number of[[(_ FloatingPoint eb sb)]];
  //    [[(_ to_fp_unsigned eb sb)]](r, x) = y otherwise, where y is the finite number
  //    such that[[fp.to_real]](y) is closest to n according to rounding mode r.

  unsigned ebits = to->get_exponent_width();
  unsigned sbits = to->get_significand_width();
  unsigned bv_sz = x->sort->get_data_width();

  smt_astt bv0_1 = ctx->mk_smt_bv(BigInt(0), 1);
  smt_astt bv0_sz = ctx->mk_smt_bv(BigInt(0), bv_sz);

  smt_astt is_zero = ctx->mk_eq(x, bv0_sz);

  smt_astt pzero = mk_pzero(ebits, sbits);

  // Special case: x == 0 -> p/n zero
  smt_astt c1 = is_zero;
  smt_astt v1 = pzero;

  // Special case: x != 0
  // x is [bv_sz-1] . [bv_sz-2 ... 0] * 2^(bv_sz-1)
  // bv_sz-1 is the "1.0" bit for the rounder.

  smt_astt lz = mk_leading_zeros(x, bv_sz);
  smt_astt shifted_sig = ctx->mk_bvshl(x, lz);

  // shifted_sig is [bv_sz-1] . [bv_sz-2 ... 0] * 2^(bv_sz-1) * 2^(-lz)
  unsigned sig_sz = sbits + 4; // we want extra rounding bits.

  smt_astt sig_4, sticky;
  if(sig_sz <= bv_sz)
  {
    // one short
    sig_4 = ctx->mk_extract(shifted_sig, bv_sz - 1, bv_sz - sig_sz + 1);

    smt_astt sig_rest = ctx->mk_extract(shifted_sig, bv_sz - sig_sz, 0);
    sticky = ctx->mk_bvredor(sig_rest);
    sig_4 = ctx->mk_concat(sig_4, sticky);
  }
  else
  {
    unsigned extra_bits = sig_sz - bv_sz;
    smt_astt extra_zeros = ctx->mk_smt_bv(BigInt(0), extra_bits);
    sig_4 = ctx->mk_concat(shifted_sig, extra_zeros);
    lz = ctx->mk_bvadd(
      ctx->mk_concat(extra_zeros, lz),
      ctx->mk_smt_bv(BigInt(extra_bits), sig_sz));
    bv_sz = bv_sz + extra_bits;
  }
  assert(sig_4->sort->get_data_width() == sig_sz);

  smt_astt s_exp = ctx->mk_bvsub(ctx->mk_smt_bv(BigInt(bv_sz - 2), bv_sz), lz);

  // s_exp = (bv_sz-2) + (-lz) signed
  assert(s_exp->sort->get_data_width() == bv_sz);

  unsigned exp_sz = ebits + 2; // (+2 for rounder)
  smt_astt exp_2 = ctx->mk_extract(s_exp, exp_sz - 1, 0);

  // the remaining bits are 0 if ebits is large enough.
  smt_astt exp_too_large = ctx->mk_smt_bool(false); // This is always in range.

  // The exponent is at most bv_sz, i.e., we need ld(bv_sz)+1 ebits.
  // exp < bv_sz (+sign bit which is [0])
  unsigned exp_worst_case_sz =
    (unsigned)((log((double)bv_sz) / log((double)2)) + 1.0);

  if(exp_sz < exp_worst_case_sz)
  {
    // exp_sz < exp_worst_case_sz and exp >= 0.
    // Take the maximum legal exponent; this
    // allows us to keep the most precision.
    smt_astt max_exp = mk_max_exp(exp_sz);
    smt_astt max_exp_bvsz = ctx->mk_zero_ext(max_exp, bv_sz - exp_sz);

    exp_too_large = ctx->mk_bvule(
      ctx->mk_bvadd(max_exp_bvsz, ctx->mk_smt_bv(BigInt(1), bv_sz)), s_exp);
    smt_astt zero_sig_sz = ctx->mk_smt_bv(BigInt(0), sig_sz);
    sig_4 = ctx->mk_ite(exp_too_large, zero_sig_sz, sig_4);
    exp_2 = ctx->mk_ite(exp_too_large, max_exp, exp_2);
  }

  smt_astt sgn, sig, exp;
  sgn = bv0_1;
  sig = sig_4;
  exp = exp_2;

  assert(sig->sort->get_data_width() == sbits + 4);
  assert(exp->sort->get_data_width() == ebits + 2);

  smt_astt v2;
  round(rm, sgn, sig, exp, ebits, sbits, v2);

  return ctx->mk_ite(c1, v1, v2);
}

smt_astt
fp_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt x, smt_sortt to, smt_astt rm)
{
  // This is a conversion from unsigned bitvector to float:
  // ((_ to_fp_unsigned eb sb) RoundingMode (_ BitVec m) (_ FloatingPoint eb sb))
  // Semantics:
  //    Let b in[[(_ BitVec m)]] and let n be the unsigned integer represented by b.
  //    [[(_ to_fp_unsigned eb sb)]](r, x) = +infinity if n is too large to be
  //    represented as a finite number of[[(_ FloatingPoint eb sb)]];
  //    [[(_ to_fp_unsigned eb sb)]](r, x) = y otherwise, where y is the finite number
  //    such that[[fp.to_real]](y) is closest to n according to rounding mode r.

  unsigned ebits = to->get_exponent_width();
  unsigned sbits = to->get_significand_width();
  unsigned bv_sz = x->sort->get_data_width();

  smt_astt bv1_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt bv0_sz = ctx->mk_smt_bv(BigInt(0), bv_sz);

  smt_astt is_zero = ctx->mk_eq(x, bv0_sz);

  smt_astt pzero = mk_pzero(ebits, sbits);

  // Special case: x == 0 -> p/n zero
  smt_astt c1 = is_zero;
  smt_astt v1 = pzero;

  // Special case: x != 0
  smt_astt is_neg_bit = ctx->mk_extract(x, bv_sz - 1, bv_sz - 1);
  smt_astt is_neg = ctx->mk_eq(is_neg_bit, bv1_1);
  smt_astt neg_x = ctx->mk_bvneg(x);
  smt_astt x_abs = ctx->mk_ite(is_neg, neg_x, x);

  // x is [bv_sz-1] . [bv_sz-2 ... 0] * 2^(bv_sz-1)
  // bv_sz-1 is the "1.0" bit for the rounder.

  smt_astt lz = mk_leading_zeros(x_abs, bv_sz);
  smt_astt shifted_sig = ctx->mk_bvshl(x_abs, lz);

  // shifted_sig is [bv_sz-1, bv_sz-2] . [bv_sz-3 ... 0] * 2^(bv_sz-2) * 2^(-lz)
  unsigned sig_sz = sbits + 4; // we want extra rounding bits.

  smt_astt sig_4, sticky;
  if(sig_sz <= bv_sz)
  {
    // one short
    sig_4 = ctx->mk_extract(shifted_sig, bv_sz - 1, bv_sz - sig_sz + 1);

    smt_astt sig_rest = ctx->mk_extract(shifted_sig, bv_sz - sig_sz, 0);
    sticky = ctx->mk_bvredor(sig_rest);
    sig_4 = ctx->mk_concat(sig_4, sticky);
  }
  else
  {
    unsigned extra_bits = sig_sz - bv_sz;
    smt_astt extra_zeros = ctx->mk_smt_bv(BigInt(0), extra_bits);
    sig_4 = ctx->mk_concat(shifted_sig, extra_zeros);
    lz = ctx->mk_bvadd(
      ctx->mk_concat(extra_zeros, lz),
      ctx->mk_smt_bv(BigInt(extra_bits), sig_sz));
    bv_sz = bv_sz + extra_bits;
  }
  assert(sig_4->sort->get_data_width() == sig_sz);

  smt_astt s_exp = ctx->mk_bvsub(ctx->mk_smt_bv(BigInt(bv_sz - 2), bv_sz), lz);

  // s_exp = (bv_sz-2) + (-lz) signed
  assert(s_exp->sort->get_data_width() == bv_sz);

  unsigned exp_sz = ebits + 2; // (+2 for rounder)
  smt_astt exp_2 = ctx->mk_extract(s_exp, exp_sz - 1, 0);

  // the remaining bits are 0 if ebits is large enough.
  smt_astt exp_too_large = ctx->mk_smt_bool(false);

  // The exponent is at most bv_sz, i.e., we need ld(bv_sz)+1 ebits.
  // exp < bv_sz (+sign bit which is [0])
  unsigned exp_worst_case_sz =
    (unsigned)((log((double)bv_sz) / log((double)2)) + 1.0);

  if(exp_sz < exp_worst_case_sz)
  {
    // exp_sz < exp_worst_case_sz and exp >= 0.
    // Take the maximum legal exponent; this
    // allows us to keep the most precision.
    smt_astt max_exp = mk_max_exp(exp_sz);
    smt_astt max_exp_bvsz = ctx->mk_zero_ext(max_exp, bv_sz - exp_sz);

    exp_too_large = ctx->mk_bvule(
      ctx->mk_bvadd(max_exp_bvsz, ctx->mk_smt_bv(BigInt(1), bv_sz)), s_exp);
    smt_astt zero_sig_sz = ctx->mk_smt_bv(BigInt(0), sig_sz);
    sig_4 = ctx->mk_ite(exp_too_large, zero_sig_sz, sig_4);
    exp_2 = ctx->mk_ite(exp_too_large, max_exp, exp_2);
  }

  smt_astt sgn, sig, exp;
  sgn = is_neg_bit;
  sig = sig_4;
  exp = exp_2;

  assert(sig->sort->get_data_width() == sbits + 4);
  assert(exp->sort->get_data_width() == ebits + 2);

  smt_astt v2;
  round(rm, sgn, sig, exp, ebits, sbits, v2);

  return ctx->mk_ite(c1, v1, v2);
}

ieee_floatt fp_convt::get_fpbv(smt_astt a)
{
  std::size_t width = a->sort->get_data_width();
  std::size_t swidth = a->sort->get_significand_width();

  ieee_floatt number(ieee_float_spect(swidth - 1, width - swidth));
  number.unpack(ctx->get_bv(a));
  return number;
}

void fp_convt::add_core(
  unsigned sbits,
  unsigned ebits,
  smt_astt &c_sgn,
  smt_astt &c_sig,
  smt_astt &c_exp,
  smt_astt &d_sgn,
  smt_astt &d_sig,
  smt_astt &d_exp,
  smt_astt &res_sgn,
  smt_astt &res_sig,
  smt_astt &res_exp)
{
  // c/d are now such that c_exp >= d_exp.
  smt_astt exp_delta = ctx->mk_bvsub(c_exp, d_exp);

  if(log2(sbits + 2) < ebits + 2)
  {
    // cap the delta
    smt_astt cap = ctx->mk_smt_bv(BigInt(sbits + 2), ebits + 2);
    smt_astt cap_le_delta = ctx->mk_bvule(cap, ctx->mk_zero_ext(exp_delta, 2));
    smt_astt exp_delta_ext = ctx->mk_zero_ext(exp_delta, 2);
    exp_delta = ctx->mk_ite(cap_le_delta, cap, exp_delta_ext);
    exp_delta = ctx->mk_extract(exp_delta, ebits - 1, 0);
  }

  // Three extra bits for c/d
  c_sig = ctx->mk_concat(c_sig, ctx->mk_smt_bv(BigInt(0), 3));
  d_sig = ctx->mk_concat(d_sig, ctx->mk_smt_bv(BigInt(0), 3));

  // Alignment shift with sticky bit computation.
  smt_astt big_d_sig =
    ctx->mk_concat(d_sig, ctx->mk_smt_bv(BigInt(0), sbits + 3));

  smt_astt shifted_big = ctx->mk_bvlshr(
    big_d_sig,
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(0), (2 * (sbits + 3)) - ebits), exp_delta));
  smt_astt shifted_d_sig =
    ctx->mk_extract(shifted_big, (2 * (sbits + 3) - 1), (sbits + 3));

  smt_astt sticky_raw = ctx->mk_extract(shifted_big, sbits + 2, 0);
  smt_astt nil_sbit3 = ctx->mk_smt_bv(BigInt(0), sbits + 3);
  smt_astt one_sbit3 = ctx->mk_smt_bv(BigInt(1), sbits + 3);
  smt_astt sticky_eq = ctx->mk_eq(sticky_raw, nil_sbit3);
  smt_astt sticky = ctx->mk_ite(sticky_eq, nil_sbit3, one_sbit3);

  shifted_d_sig = ctx->mk_bvor(shifted_d_sig, sticky);

  smt_astt eq_sgn = ctx->mk_eq(c_sgn, d_sgn);

  // two extra bits for catching the overflow.
  c_sig = ctx->mk_zero_ext(c_sig, 2);
  shifted_d_sig = ctx->mk_zero_ext(shifted_d_sig, 2);

  assert(c_sig->sort->get_data_width() == sbits + 5);
  assert(shifted_d_sig->sort->get_data_width() == sbits + 5);

  smt_astt c_plus_d = ctx->mk_bvadd(c_sig, shifted_d_sig);
  smt_astt c_minus_d = ctx->mk_bvsub(c_sig, shifted_d_sig);
  smt_astt sum = ctx->mk_ite(eq_sgn, c_plus_d, c_minus_d);

  smt_astt sign_bv = ctx->mk_extract(sum, sbits + 4, sbits + 4);
  smt_astt n_sum = ctx->mk_bvneg(sum);

  smt_astt not_c_sgn = ctx->mk_bvnot(c_sgn);
  smt_astt not_d_sgn = ctx->mk_bvnot(d_sgn);
  smt_astt not_sign_bv = ctx->mk_bvnot(sign_bv);
  smt_astt res_sgn_c1 = ctx->mk_bvand(ctx->mk_bvand(not_c_sgn, d_sgn), sign_bv);
  smt_astt res_sgn_c2 =
    ctx->mk_bvand(ctx->mk_bvand(c_sgn, not_d_sgn), not_sign_bv);
  smt_astt res_sgn_c3 = ctx->mk_bvand(c_sgn, d_sgn);
  res_sgn = ctx->mk_bvor(ctx->mk_bvor(res_sgn_c1, res_sgn_c2), res_sgn_c3);

  smt_astt one_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt res_sig_eq = ctx->mk_eq(sign_bv, one_1);
  smt_astt sig_abs = ctx->mk_ite(res_sig_eq, n_sum, sum);

  res_sig = ctx->mk_extract(sig_abs, sbits + 3, 0);
  res_exp = ctx->mk_sign_ext(c_exp, 2); // rounder requires 2 extra bits!
}

smt_astt fp_convt::mk_smt_fpbv_add(smt_astt x, smt_astt y, smt_astt rm)
{
  assert(x->sort->get_data_width() == y->sort->get_data_width());
  assert(x->sort->get_exponent_width() == y->sort->get_exponent_width());

  std::size_t ebits = x->sort->get_exponent_width();
  std::size_t sbits = x->sort->get_significand_width();

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);
  smt_astt nzero = mk_nzero(ebits, sbits);
  smt_astt pzero = mk_pzero(ebits, sbits);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt x_is_neg = mk_is_neg(x);
  smt_astt x_is_inf = mk_smt_fpbv_is_inf(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt y_is_neg = mk_is_neg(y);
  smt_astt y_is_inf = mk_smt_fpbv_is_inf(y);

  smt_astt c1 = ctx->mk_or(x_is_nan, y_is_nan);
  smt_astt v1 = nan;

  smt_astt c2 = mk_smt_fpbv_is_inf(x);
  smt_astt nx = mk_is_neg(x);
  smt_astt ny = mk_is_neg(y);
  smt_astt nx_xor_ny = ctx->mk_xor(nx, ny);
  smt_astt inf_xor = ctx->mk_and(y_is_inf, nx_xor_ny);
  smt_astt v2 = ctx->mk_ite(inf_xor, nan, x);

  smt_astt c3 = mk_smt_fpbv_is_inf(y);
  smt_astt xy_is_neg = ctx->mk_xor(x_is_neg, y_is_neg);
  smt_astt v3_and = ctx->mk_and(x_is_inf, xy_is_neg);
  smt_astt v3 = ctx->mk_ite(v3_and, nan, y);

  smt_astt c4 = ctx->mk_and(x_is_zero, y_is_zero);
  smt_astt signs_and = ctx->mk_and(x_is_neg, y_is_neg);
  smt_astt signs_xor = ctx->mk_xor(x_is_neg, y_is_neg);
  smt_astt rm_is_to_neg = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt rm_and_xor = ctx->mk_and(rm_is_to_neg, signs_xor);
  smt_astt neg_cond = ctx->mk_or(signs_and, rm_and_xor);
  smt_astt v4 = ctx->mk_ite(neg_cond, nzero, pzero);
  smt_astt v4_and = ctx->mk_and(x_is_neg, y_is_neg);
  v4 = ctx->mk_ite(v4_and, x, v4);

  smt_astt c5 = x_is_zero;
  smt_astt v5 = y;

  smt_astt c6 = y_is_zero;
  smt_astt v6 = x;

  // Actual addition.
  smt_astt a_sgn, a_sig, a_exp, a_lz, b_sgn, b_sig, b_exp, b_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, false);
  unpack(y, b_sgn, b_sig, b_exp, b_lz, false);

  smt_astt swap_cond = ctx->mk_bvsle(a_exp, b_exp);

  smt_astt c_sgn = ctx->mk_ite(swap_cond, b_sgn, a_sgn);
  smt_astt c_sig = ctx->mk_ite(swap_cond, b_sig, a_sig); // has sbits
  smt_astt c_exp = ctx->mk_ite(swap_cond, b_exp, a_exp); // has ebits
  smt_astt d_sgn = ctx->mk_ite(swap_cond, a_sgn, b_sgn);
  smt_astt d_sig = ctx->mk_ite(swap_cond, a_sig, b_sig); // has sbits
  smt_astt d_exp = ctx->mk_ite(swap_cond, a_exp, b_exp); // has ebits

  smt_astt res_sgn, res_sig, res_exp;
  add_core(
    sbits,
    ebits,
    c_sgn,
    c_sig,
    c_exp,
    d_sgn,
    d_sig,
    d_exp,
    res_sgn,
    res_sig,
    res_exp);

  smt_astt nil_sbit4 = ctx->mk_smt_bv(BigInt(0), sbits + 4);
  smt_astt is_zero_sig = ctx->mk_eq(res_sig, nil_sbit4);

  smt_astt zero_case = ctx->mk_ite(rm_is_to_neg, nzero, pzero);

  smt_astt rounded;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, rounded);

  smt_astt v7 = ctx->mk_ite(is_zero_sig, zero_case, rounded);

  smt_astt result = ctx->mk_ite(c6, v6, v7);
  result = ctx->mk_ite(c5, v5, result);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt fp_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  smt_astt t = mk_smt_fpbv_neg(rhs);
  return mk_smt_fpbv_add(lhs, t, rm);
}

smt_astt fp_convt::mk_smt_fpbv_mul(smt_astt x, smt_astt y, smt_astt rm)
{
  assert(x->sort->get_data_width() == y->sort->get_data_width());
  assert(x->sort->get_exponent_width() == y->sort->get_exponent_width());

  std::size_t ebits = x->sort->get_exponent_width();
  std::size_t sbits = x->sort->get_significand_width();

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);
  smt_astt nzero = mk_nzero(ebits, sbits);
  smt_astt pzero = mk_pzero(ebits, sbits);
  smt_astt ninf = mk_ninf(ebits, sbits);
  smt_astt pinf = mk_pinf(ebits, sbits);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt x_is_pos = mk_is_pos(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt y_is_pos = mk_is_pos(y);

  // (x is NaN) || (y is NaN) -> NaN
  smt_astt c1 = ctx->mk_or(x_is_nan, y_is_nan);
  smt_astt v1 = nan;

  // (x is +oo) -> if (y is 0) then NaN else inf with y's sign.
  smt_astt c2 = mk_is_pinf(x);
  smt_astt y_sgn_inf = ctx->mk_ite(y_is_pos, pinf, ninf);
  smt_astt v2 = ctx->mk_ite(y_is_zero, nan, y_sgn_inf);

  // (y is +oo) -> if (x is 0) then NaN else inf with x's sign.
  smt_astt c3 = mk_is_pinf(y);
  smt_astt x_sgn_inf = ctx->mk_ite(x_is_pos, pinf, ninf);
  smt_astt v3 = ctx->mk_ite(x_is_zero, nan, x_sgn_inf);

  // (x is -oo) -> if (y is 0) then NaN else inf with -y's sign.
  smt_astt c4 = mk_is_ninf(x);
  smt_astt neg_y_sgn_inf = ctx->mk_ite(y_is_pos, ninf, pinf);
  smt_astt v4 = ctx->mk_ite(y_is_zero, nan, neg_y_sgn_inf);

  // (y is -oo) -> if (x is 0) then NaN else inf with -x's sign.
  smt_astt c5 = mk_is_ninf(y);
  smt_astt neg_x_sgn_inf = ctx->mk_ite(x_is_pos, ninf, pinf);
  smt_astt v5 = ctx->mk_ite(x_is_zero, nan, neg_x_sgn_inf);

  // (x is 0) || (y is 0) -> x but with sign = x.sign ^ y.sign
  smt_astt c6 = ctx->mk_or(x_is_zero, y_is_zero);
  smt_astt sign_xor = ctx->mk_xor(x_is_pos, y_is_pos);
  smt_astt v6 = ctx->mk_ite(sign_xor, nzero, pzero);

  // else comes the actual multiplication.
  smt_astt a_sgn, a_sig, a_exp, a_lz, b_sgn, b_sig, b_exp, b_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, true);
  unpack(y, b_sgn, b_sig, b_exp, b_lz, true);

  smt_astt a_lz_ext = ctx->mk_zero_ext(a_lz, 2);
  smt_astt b_lz_ext = ctx->mk_zero_ext(b_lz, 2);

  smt_astt a_sig_ext = ctx->mk_zero_ext(a_sig, sbits);
  smt_astt b_sig_ext = ctx->mk_zero_ext(b_sig, sbits);

  smt_astt a_exp_ext = ctx->mk_sign_ext(a_exp, 2);
  smt_astt b_exp_ext = ctx->mk_sign_ext(b_exp, 2);

  smt_astt res_sgn, res_sig, res_exp;
  res_sgn = ctx->mk_bvxor(a_sgn, b_sgn);

  res_exp = ctx->mk_bvadd(
    ctx->mk_bvsub(a_exp_ext, a_lz_ext), ctx->mk_bvsub(b_exp_ext, b_lz_ext));

  smt_astt product = ctx->mk_bvmul(a_sig_ext, b_sig_ext);

  assert(product->sort->get_data_width() == 2 * sbits);

  smt_astt h_p = ctx->mk_extract(product, 2 * sbits - 1, sbits);
  smt_astt l_p = ctx->mk_extract(product, sbits - 1, 0);

  smt_astt rbits;
  if(sbits >= 4)
  {
    smt_astt sticky = ctx->mk_bvredor(ctx->mk_extract(product, sbits - 4, 0));
    rbits =
      ctx->mk_concat(ctx->mk_extract(product, sbits - 1, sbits - 3), sticky);
  }
  else
    rbits = ctx->mk_concat(l_p, ctx->mk_smt_bv(BigInt(0), 4 - sbits));

  assert(rbits->sort->get_data_width() == 4);
  res_sig = ctx->mk_concat(h_p, rbits);

  smt_astt v7;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, v7);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c6, v6, v7);
  result = ctx->mk_ite(c5, v5, result);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt fp_convt::mk_smt_fpbv_div(smt_astt x, smt_astt y, smt_astt rm)
{
  assert(x->sort->get_data_width() == y->sort->get_data_width());
  assert(x->sort->get_exponent_width() == y->sort->get_exponent_width());

  unsigned ebits = x->sort->get_exponent_width();
  unsigned sbits = x->sort->get_significand_width();

  smt_astt nan = mk_smt_fpbv_nan(ebits, sbits);
  smt_astt nzero = mk_nzero(ebits, sbits);
  smt_astt pzero = mk_pzero(ebits, sbits);
  smt_astt ninf = mk_ninf(ebits, sbits);
  smt_astt pinf = mk_pinf(ebits, sbits);

  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt x_is_pos = mk_is_pos(x);
  smt_astt x_is_inf = mk_smt_fpbv_is_inf(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt y_is_pos = mk_is_pos(y);
  smt_astt y_is_inf = mk_smt_fpbv_is_inf(y);

  // (x is NaN) || (y is NaN) -> NaN
  smt_astt c1 = ctx->mk_or(x_is_nan, y_is_nan);
  smt_astt v1 = nan;

  // (x is +oo) -> if (y is oo) then NaN else inf with y's sign.
  smt_astt c2 = mk_is_pinf(x);
  smt_astt y_sgn_inf = ctx->mk_ite(y_is_pos, pinf, ninf);
  smt_astt v2 = ctx->mk_ite(y_is_inf, nan, y_sgn_inf);

  // (y is +oo) -> if (x is oo) then NaN else 0 with sign x.sgn ^ y.sgn
  smt_astt c3 = mk_is_pinf(y);
  smt_astt signs_xor = ctx->mk_xor(x_is_pos, y_is_pos);
  smt_astt xy_zero = ctx->mk_ite(signs_xor, nzero, pzero);
  smt_astt v3 = ctx->mk_ite(x_is_inf, nan, xy_zero);

  // (x is -oo) -> if (y is oo) then NaN else inf with -y's sign.
  smt_astt c4 = mk_is_ninf(x);
  smt_astt neg_y_sgn_inf = ctx->mk_ite(y_is_pos, ninf, pinf);
  smt_astt v4 = ctx->mk_ite(y_is_inf, nan, neg_y_sgn_inf);

  // (y is -oo) -> if (x is oo) then NaN else 0 with sign x.sgn ^ y.sgn
  smt_astt c5 = mk_is_ninf(y);
  smt_astt v5 = ctx->mk_ite(x_is_inf, nan, xy_zero);

  // (y is 0) -> if (x is 0) then NaN else inf with xor sign.
  smt_astt c6 = y_is_zero;
  smt_astt sgn_inf = ctx->mk_ite(signs_xor, ninf, pinf);
  smt_astt v6 = ctx->mk_ite(x_is_zero, nan, sgn_inf);

  // (x is 0) -> result is zero with sgn = x.sgn^y.sgn
  // This is a special case to avoid problems with the unpacking of zero.
  smt_astt c7 = x_is_zero;
  smt_astt v7 = ctx->mk_ite(signs_xor, nzero, pzero);

  // else comes the actual division.
  assert(ebits <= sbits);

  smt_astt a_sgn, a_sig, a_exp, a_lz, b_sgn, b_sig, b_exp, b_lz;
  unpack(x, a_sgn, a_sig, a_exp, a_lz, true);
  unpack(y, b_sgn, b_sig, b_exp, b_lz, true);

  unsigned extra_bits = sbits + 2;
  smt_astt a_sig_ext =
    ctx->mk_concat(a_sig, ctx->mk_smt_bv(BigInt(0), sbits + extra_bits));
  smt_astt b_sig_ext = ctx->mk_zero_ext(b_sig, sbits + extra_bits);

  smt_astt a_exp_ext = ctx->mk_sign_ext(a_exp, 2);
  smt_astt b_exp_ext = ctx->mk_sign_ext(b_exp, 2);

  smt_astt res_sgn = ctx->mk_bvxor(a_sgn, b_sgn);

  smt_astt a_lz_ext = ctx->mk_zero_ext(a_lz, 2);
  smt_astt b_lz_ext = ctx->mk_zero_ext(b_lz, 2);

  smt_astt res_exp = ctx->mk_bvsub(
    ctx->mk_bvsub(a_exp_ext, a_lz_ext), ctx->mk_bvsub(b_exp_ext, b_lz_ext));

  // b_sig_ext can't be 0 here, so it's safe to use OP_BUDIV_I
  smt_astt quotient = ctx->mk_bvudiv(a_sig_ext, b_sig_ext);

  assert(quotient->sort->get_data_width() == (sbits + sbits + extra_bits));

  smt_astt sticky =
    ctx->mk_bvredor(ctx->mk_extract(quotient, extra_bits - 2, 0));
  smt_astt res_sig = ctx->mk_concat(
    ctx->mk_extract(quotient, extra_bits + sbits + 1, extra_bits - 1), sticky);

  assert(res_sig->sort->get_data_width() == (sbits + 4));

  smt_astt res_sig_lz = mk_leading_zeros(res_sig, sbits + 4);
  smt_astt res_sig_shift_amount =
    ctx->mk_bvsub(res_sig_lz, ctx->mk_smt_bv(BigInt(1), sbits + 4));
  smt_astt shift_cond =
    ctx->mk_bvule(res_sig_lz, ctx->mk_smt_bv(BigInt(1), sbits + 4));
  smt_astt res_sig_shifted = ctx->mk_bvshl(res_sig, res_sig_shift_amount);
  smt_astt res_exp_shifted =
    ctx->mk_bvsub(res_exp, ctx->mk_extract(res_sig_shift_amount, ebits + 1, 0));
  res_sig = ctx->mk_ite(shift_cond, res_sig, res_sig_shifted);
  res_exp = ctx->mk_ite(shift_cond, res_exp, res_exp_shifted);

  smt_astt v8;
  round(rm, res_sgn, res_sig, res_exp, ebits, sbits, v8);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c7, v7, v8);
  result = ctx->mk_ite(c6, v6, result);
  result = ctx->mk_ite(c5, v5, result);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  return ctx->mk_ite(c1, v1, result);
}

smt_astt fp_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  // +0 and -0 should return true
  smt_astt is_zero0 = mk_smt_fpbv_is_zero(lhs);
  smt_astt is_zero1 = mk_smt_fpbv_is_zero(rhs);
  smt_astt both_zero = ctx->mk_and(is_zero0, is_zero1);

  // Check if they are NaN
  smt_astt isnan0 = mk_smt_fpbv_is_nan(lhs);
  smt_astt isnan1 = mk_smt_fpbv_is_nan(rhs);
  smt_astt nan = ctx->mk_or(isnan0, isnan1);

  // Otherwise compare them bitwise
  smt_astt are_equal = ctx->mk_eq(lhs, rhs);

  // They are equal if they are either +0 and -0 (and vice-versa) or bitwise
  // equal and neither is NaN
  return ctx->mk_and(ctx->mk_or(both_zero, are_equal), ctx->mk_not(nan));
}

smt_astt fp_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  // (a > b) iff (b < a)
  return mk_smt_fpbv_lt(rhs, lhs);
}

smt_astt fp_convt::mk_smt_fpbv_lt(smt_astt x, smt_astt y)
{
  smt_astt x_is_nan = mk_smt_fpbv_is_nan(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt c1 = ctx->mk_or(x_is_nan, y_is_nan);
  smt_astt x_is_zero = mk_smt_fpbv_is_zero(x);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt c2 = ctx->mk_and(x_is_zero, y_is_zero);

  smt_astt x_sgn = extract_signbit(ctx, x);
  smt_astt x_sig = extract_significand(ctx, x);
  smt_astt x_exp = extract_exponent(ctx, x);

  smt_astt y_sgn = extract_signbit(ctx, y);
  smt_astt y_sig = extract_significand(ctx, y);
  smt_astt y_exp = extract_exponent(ctx, y);

  smt_astt one_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt nil_1 = ctx->mk_smt_bv(BigInt(0), 1);
  smt_astt c3 = ctx->mk_eq(x_sgn, one_1);

  smt_astt y_sgn_eq_0 = ctx->mk_eq(y_sgn, nil_1);
  smt_astt y_lt_x_exp = ctx->mk_bvult(y_exp, x_exp);
  smt_astt y_lt_x_sig = ctx->mk_bvult(y_sig, x_sig);
  smt_astt y_eq_x_exp = ctx->mk_eq(y_exp, x_exp);
  smt_astt y_le_x_sig_exp = ctx->mk_and(y_eq_x_exp, y_lt_x_sig);
  smt_astt t3_or = ctx->mk_or(y_lt_x_exp, y_le_x_sig_exp);
  smt_astt t3 = ctx->mk_ite(y_sgn_eq_0, ctx->mk_smt_bool(true), t3_or);

  smt_astt y_sgn_eq_1 = ctx->mk_eq(y_sgn, one_1);
  smt_astt x_lt_y_exp = ctx->mk_bvult(x_exp, y_exp);
  smt_astt x_eq_y_exp = ctx->mk_eq(x_exp, y_exp);
  smt_astt x_lt_y_sig = ctx->mk_bvult(x_sig, y_sig);
  smt_astt x_le_y_sig_exp = ctx->mk_and(x_eq_y_exp, x_lt_y_sig);
  smt_astt t4_or = ctx->mk_or(x_lt_y_exp, x_le_y_sig_exp);
  smt_astt t4 = ctx->mk_ite(y_sgn_eq_1, ctx->mk_smt_bool(false), t4_or);

  smt_astt c3t3t4 = ctx->mk_ite(c3, t3, t4);
  smt_astt c2else = ctx->mk_ite(c2, ctx->mk_smt_bool(false), c3t3t4);
  return ctx->mk_ite(c1, ctx->mk_smt_bool(false), c2else);
}

smt_astt fp_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  // This is !FPLT
  smt_astt a = mk_smt_fpbv_lt(lhs, rhs);
  return ctx->mk_not(a);
}

smt_astt fp_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  smt_astt lt = mk_smt_fpbv_lt(lhs, rhs);
  smt_astt eq = mk_smt_fpbv_eq(lhs, rhs);
  return ctx->mk_or(lt, eq);
}

smt_astt fp_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  // Extract the exponent and significand
  smt_astt exp = extract_exponent(ctx, op);
  smt_astt sig = extract_significand(ctx, op);

  // exp == 1^n , sig != 0
  smt_astt top_exp = mk_top_exp(exp->sort->get_data_width());

  smt_astt zero = ctx->mk_smt_bv(BigInt(0), sig->sort->get_data_width());
  smt_astt sig_is_zero = ctx->mk_eq(sig, zero);
  smt_astt sig_is_not_zero = ctx->mk_not(sig_is_zero);
  smt_astt exp_is_top = ctx->mk_eq(exp, top_exp);
  return ctx->mk_and(exp_is_top, sig_is_not_zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  // Extract the exponent and significand
  smt_astt exp = extract_exponent(ctx, op);
  smt_astt sig = extract_significand(ctx, op);

  // exp == 1^n , sig == 0
  smt_astt top_exp = mk_top_exp(exp->sort->get_data_width());

  smt_astt zero = ctx->mk_smt_bv(BigInt(0), sig->sort->get_data_width());
  smt_astt sig_is_zero = ctx->mk_eq(sig, zero);
  smt_astt exp_is_top = ctx->mk_eq(exp, top_exp);
  return ctx->mk_and(exp_is_top, sig_is_zero);
}

smt_astt fp_convt::mk_is_denormal(smt_astt op)
{
  // Extract the exponent and significand
  smt_astt exp = extract_exponent(ctx, op);

  smt_astt zero = ctx->mk_smt_bv(BigInt(0), exp->sort->get_data_width());
  smt_astt zexp = ctx->mk_eq(exp, zero);
  smt_astt is_zero = mk_smt_fpbv_is_zero(op);
  smt_astt n_is_zero = ctx->mk_not(is_zero);
  return ctx->mk_and(n_is_zero, zexp);
}

smt_astt fp_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  // Extract the exponent and significand
  smt_astt exp = extract_exponent(ctx, op);

  smt_astt is_denormal = mk_is_denormal(op);
  smt_astt is_zero = mk_smt_fpbv_is_zero(op);

  unsigned ebits = exp->sort->get_data_width();
  smt_astt p = ctx->mk_smt_bv(BigInt(power2m1(ebits, false)), ebits);

  smt_astt is_special = ctx->mk_eq(exp, p);

  smt_astt or_ex = ctx->mk_or(is_special, is_denormal);
  or_ex = ctx->mk_or(is_zero, or_ex);
  return ctx->mk_not(or_ex);
}

smt_astt fp_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  // Both -0 and 0 should return true

  // Compare with '0'
  smt_astt zero = ctx->mk_smt_bv(BigInt(0), op->sort->get_data_width() - 1);

  // Extract everything but the sign bit
  smt_astt ew_sw = extract_exp_sig(ctx, op);

  return ctx->mk_eq(ew_sw, zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  smt_astt t1 = mk_smt_fpbv_is_nan(op);
  smt_astt t2 = mk_is_neg(op);
  smt_astt nt1 = ctx->mk_not(t1);
  return ctx->mk_and(nt1, t2);
}

smt_astt fp_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  smt_astt t1 = mk_smt_fpbv_is_nan(op);
  smt_astt t2 = mk_is_pos(op);
  smt_astt nt1 = ctx->mk_not(t1);
  return ctx->mk_and(nt1, t2);
}

smt_astt fp_convt::mk_smt_fpbv_abs(smt_astt op)
{
  // Extract everything but the sign bit
  smt_astt ew_sw = extract_exp_sig(ctx, op);

  // Concat that with '0'
  smt_astt zero = ctx->mk_smt_bv(BigInt(0), 1);
  return mk_from_bv_to_fp(ctx->mk_concat(zero, ew_sw), op->sort);
}

smt_astt fp_convt::mk_smt_fpbv_neg(smt_astt op)
{
  // Extract everything but the sign bit
  smt_astt ew_sw = extract_exp_sig(ctx, op);
  smt_astt sgn = extract_signbit(ctx, op);

  smt_astt c = mk_smt_fpbv_is_nan(op);
  smt_astt nsgn = ctx->mk_bvnot(sgn);
  smt_astt r_sgn = ctx->mk_ite(c, sgn, nsgn);
  return mk_from_bv_to_fp(ctx->mk_concat(r_sgn, ew_sw), op->sort);
}

void fp_convt::unpack(
  smt_astt &src,
  smt_astt &sgn,
  smt_astt &sig,
  smt_astt &exp,
  smt_astt &lz,
  bool normalize)
{
  unsigned sbits = src->sort->get_significand_width();
  unsigned ebits = src->sort->get_exponent_width();

  // Extract parts
  sgn = extract_signbit(ctx, src);
  exp = extract_exponent(ctx, src);
  sig = extract_significand(ctx, src);

  assert(sgn->sort->get_data_width() == 1);
  assert(exp->sort->get_data_width() == ebits);
  assert(sig->sort->get_data_width() == sbits - 1);

  smt_astt is_normal = mk_smt_fpbv_is_normal(src);
  smt_astt normal_sig = ctx->mk_concat(ctx->mk_smt_bv(BigInt(1), 1), sig);
  smt_astt normal_exp = mk_unbias(exp);

  smt_astt denormal_sig = ctx->mk_zero_ext(sig, 1);
  smt_astt denormal_exp = ctx->mk_smt_bv(BigInt(1), ebits);
  denormal_exp = mk_unbias(denormal_exp);

  smt_astt zero_e = ctx->mk_smt_bv(BigInt(0), ebits);
  if(normalize)
  {
    smt_astt zero_s = ctx->mk_smt_bv(BigInt(0), sbits);
    smt_astt is_sig_zero = ctx->mk_eq(zero_s, denormal_sig);

    smt_astt lz_d = mk_leading_zeros(denormal_sig, ebits);

    smt_astt norm_or_zero = ctx->mk_or(is_normal, is_sig_zero);
    lz = ctx->mk_ite(norm_or_zero, zero_e, lz_d);

    smt_astt shift = ctx->mk_ite(is_sig_zero, zero_e, lz);
    assert(shift->sort->get_data_width() == ebits);
    if(ebits <= sbits)
    {
      smt_astt q = ctx->mk_zero_ext(shift, sbits - ebits);
      denormal_sig = ctx->mk_bvshl(denormal_sig, q);
    }
    else
    {
      // the maximum shift is `sbits', because after that the mantissa
      // would be zero anyways. So we can safely cut the shift variable down,
      // as long as we check the higher bits.
      smt_astt zero_ems = ctx->mk_smt_bv(BigInt(0), ebits - sbits);
      smt_astt sbits_s = ctx->mk_smt_bv(BigInt(sbits), sbits);
      smt_astt sh = ctx->mk_extract(shift, ebits - 1, sbits);
      smt_astt is_sh_zero = ctx->mk_eq(zero_ems, sh);
      smt_astt short_shift = ctx->mk_extract(shift, sbits - 1, 0);
      smt_astt sl = ctx->mk_ite(is_sh_zero, short_shift, sbits_s);
      denormal_sig = ctx->mk_bvshl(denormal_sig, sl);
    }
  }
  else
    lz = zero_e;

  sig = ctx->mk_ite(is_normal, normal_sig, denormal_sig);
  exp = ctx->mk_ite(is_normal, normal_exp, denormal_exp);

  assert(sgn->sort->get_data_width() == 1);
  assert(sig->sort->get_data_width() == sbits);
  assert(exp->sort->get_data_width() == ebits);
}

smt_astt fp_convt::mk_unbias(smt_astt &src)
{
  unsigned ebits = src->sort->get_data_width();

  smt_astt e_plus_one = ctx->mk_bvadd(src, ctx->mk_smt_bv(BigInt(1), ebits));

  smt_astt leading = ctx->mk_extract(e_plus_one, ebits - 1, ebits - 1);
  smt_astt n_leading = ctx->mk_bvnot(leading);
  smt_astt rest = ctx->mk_extract(e_plus_one, ebits - 2, 0);
  return ctx->mk_concat(n_leading, rest);
}

smt_astt fp_convt::mk_leading_zeros(smt_astt &src, std::size_t max_bits)
{
  std::size_t bv_sz = src->sort->get_data_width();
  if(bv_sz == 0)
    return ctx->mk_smt_bv(BigInt(0), max_bits);

  if(bv_sz == 1)
  {
    smt_astt nil_1 = ctx->mk_smt_bv(BigInt(0), 1);
    smt_astt one_m = ctx->mk_smt_bv(BigInt(1), max_bits);
    smt_astt nil_m = ctx->mk_smt_bv(BigInt(0), max_bits);

    smt_astt eq = ctx->mk_eq(src, nil_1);
    return ctx->mk_ite(eq, one_m, nil_m);
  }

  smt_astt H = ctx->mk_extract(src, bv_sz - 1, bv_sz / 2);
  smt_astt L = ctx->mk_extract(src, bv_sz / 2 - 1, 0);

  unsigned H_size = H->sort->get_data_width();

  smt_astt lzH = mk_leading_zeros(H, max_bits); /* recursive! */
  smt_astt lzL = mk_leading_zeros(L, max_bits);

  smt_astt nil_h = ctx->mk_smt_bv(BigInt(0), H_size);
  smt_astt H_is_zero = ctx->mk_eq(H, nil_h);

  smt_astt h_m = ctx->mk_smt_bv(BigInt(H_size), max_bits);
  smt_astt sum = ctx->mk_bvadd(h_m, lzL);
  return ctx->mk_ite(H_is_zero, sum, lzH);
}

void fp_convt::round(
  smt_astt &rm,
  smt_astt &sgn,
  smt_astt &sig,
  smt_astt &exp,
  unsigned ebits,
  unsigned sbits,
  smt_astt &result)
{
  // Assumptions: sig is of the form f[-1:0] . f[1:sbits-1] [guard,round,sticky],
  // i.e., it has 2 + (sbits-1) + 3 = sbits + 4 bits, where the first one is in sgn.
  // Furthermore, note that sig is an unsigned bit-vector, while exp is signed.

  assert(rm->sort->get_data_width() == 3);
  assert(sgn->sort->get_data_width() == 1);
  assert(sig->sort->get_data_width() >= 5);
  assert(exp->sort->get_data_width() >= 4);

  assert(sig->sort->get_data_width() == sbits + 4);
  assert(exp->sort->get_data_width() == ebits + 2);

  smt_astt e_min = mk_min_exp(ebits);
  smt_astt e_max = mk_max_exp(ebits);

  smt_astt one_1 = ctx->mk_smt_bv(BigInt(1), 1);
  smt_astt h_exp = ctx->mk_extract(exp, ebits + 1, ebits + 1);
  smt_astt sh_exp = ctx->mk_extract(exp, ebits, ebits);
  smt_astt th_exp = ctx->mk_extract(exp, ebits - 1, ebits - 1);
  smt_astt e3 = ctx->mk_eq(h_exp, one_1);
  smt_astt e2 = ctx->mk_eq(sh_exp, one_1);
  smt_astt e1 = ctx->mk_eq(th_exp, one_1);
  smt_astt e21 = ctx->mk_or(e2, e1);
  smt_astt ne3 = ctx->mk_not(e3);
  smt_astt e_top_three = ctx->mk_and(ne3, e21);

  smt_astt ext_emax = ctx->mk_zero_ext(e_max, 2);
  smt_astt t_sig = ctx->mk_extract(sig, sbits + 3, sbits + 3);
  smt_astt e_eq_emax = ctx->mk_eq(ext_emax, exp);
  smt_astt sigm1 = ctx->mk_eq(t_sig, one_1);
  smt_astt e_eq_emax_and_sigm1 = ctx->mk_and(e_eq_emax, sigm1);
  smt_astt OVF1 = ctx->mk_or(e_top_three, e_eq_emax_and_sigm1);

  // CMW: is this always large enough?
  smt_astt lz = mk_leading_zeros(sig, ebits + 2);

  smt_astt t = ctx->mk_bvadd(exp, ctx->mk_smt_bv(BigInt(1), ebits + 2));
  t = ctx->mk_bvsub(t, lz);
  t = ctx->mk_bvsub(t, ctx->mk_sign_ext(e_min, 2));
  smt_astt TINY =
    ctx->mk_bvsle(t, ctx->mk_smt_bv(BigInt(ULLONG_MAX), ebits + 2));

  smt_astt beta =
    ctx->mk_bvadd(ctx->mk_bvsub(exp, lz), ctx->mk_smt_bv(BigInt(1), ebits + 2));

  smt_astt sigma_add = ctx->mk_bvsub(exp, ctx->mk_sign_ext(e_min, 2));
  sigma_add = ctx->mk_bvadd(sigma_add, ctx->mk_smt_bv(BigInt(1), ebits + 2));
  smt_astt sigma = ctx->mk_ite(TINY, sigma_add, lz);

  // Normalization shift
  std::size_t sig_size = sig->sort->get_data_width();
  assert(sig_size == sbits + 4);
  assert(sigma->sort->get_data_width() == ebits + 2);
  std::size_t sigma_size = ebits + 2;

  smt_astt sigma_neg = ctx->mk_bvneg(sigma);
  smt_astt sigma_cap = ctx->mk_smt_bv(BigInt(sbits + 2), sigma_size);
  smt_astt sigma_le_cap = ctx->mk_bvule(sigma_neg, sigma_cap);
  smt_astt sigma_neg_capped = ctx->mk_ite(sigma_le_cap, sigma_neg, sigma_cap);
  smt_astt sigma_lt_zero =
    ctx->mk_bvsle(sigma, ctx->mk_smt_bv(BigInt(ULLONG_MAX), sigma_size));

  smt_astt sig_ext = ctx->mk_concat(sig, ctx->mk_smt_bv(BigInt(0), sig_size));
  smt_astt rs_sig = ctx->mk_bvlshr(
    sig_ext, ctx->mk_zero_ext(sigma_neg_capped, 2 * sig_size - sigma_size));
  smt_astt ls_sig =
    ctx->mk_bvshl(sig_ext, ctx->mk_zero_ext(sigma, 2 * sig_size - sigma_size));
  smt_astt big_sh_sig = ctx->mk_ite(sigma_lt_zero, rs_sig, ls_sig);
  assert(big_sh_sig->sort->get_data_width() == 2 * sig_size);

  std::size_t sig_extract_low_bit = (2 * sig_size - 1) - (sbits + 2) + 1;
  sig = ctx->mk_extract(big_sh_sig, 2 * sig_size - 1, sig_extract_low_bit);
  assert(sig->sort->get_data_width() == sbits + 2);

  smt_astt sticky =
    ctx->mk_bvredor(ctx->mk_extract(big_sh_sig, sig_extract_low_bit - 1, 0));

  // put the sticky bit into the significand.
  smt_astt ext_sticky = ctx->mk_zero_ext(sticky, sbits + 1);
  sig = ctx->mk_bvor(sig, ext_sticky);
  assert(sig->sort->get_data_width() == sbits + 2);

  smt_astt ext_emin = ctx->mk_zero_ext(e_min, 2);
  exp = ctx->mk_ite(TINY, ext_emin, beta);

  // Significand rounding
  sticky = ctx->mk_extract(sig, 0, 0); // new sticky bit!
  smt_astt round = ctx->mk_extract(sig, 1, 1);
  smt_astt last = ctx->mk_extract(sig, 2, 2);

  sig = ctx->mk_extract(sig, sbits + 1, 2);

  smt_astt inc = mk_rounding_decision(rm, sgn, last, round, sticky);

  assert(inc->sort->get_data_width() == 1);

  sig = ctx->mk_bvadd(ctx->mk_zero_ext(sig, 1), ctx->mk_zero_ext(inc, sbits));

  // Post normalization
  assert(sig->sort->get_data_width() == sbits + 1);
  t_sig = ctx->mk_extract(sig, sbits, sbits);
  smt_astt SIGovf = ctx->mk_eq(t_sig, one_1);

  smt_astt hallbut1_sig = ctx->mk_extract(sig, sbits, 1);
  smt_astt lallbut1_sig = ctx->mk_extract(sig, sbits - 1, 0);
  sig = ctx->mk_ite(SIGovf, hallbut1_sig, lallbut1_sig);

  assert(exp->sort->get_data_width() == ebits + 2);

  smt_astt exp_p1 = ctx->mk_bvadd(exp, ctx->mk_smt_bv(BigInt(1), ebits + 2));
  exp = ctx->mk_ite(SIGovf, exp_p1, exp);

  assert(sig->sort->get_data_width() == sbits);
  assert(exp->sort->get_data_width() == ebits + 2);
  assert(e_max->sort->get_data_width() == ebits);

  // Exponent adjustment and rounding
  smt_astt biased_exp = mk_bias(ctx->mk_extract(exp, ebits - 1, 0));

  // AdjustExp
  assert(OVF1->sort->id == SMT_SORT_BOOL);

  smt_astt exp_redand = ctx->mk_bvredand(biased_exp);
  smt_astt preOVF2 = ctx->mk_eq(exp_redand, one_1);
  smt_astt OVF2 = ctx->mk_and(SIGovf, preOVF2);
  smt_astt pem2m1 = ctx->mk_smt_bv(BigInt(power2m1(ebits - 2, false)), ebits);
  biased_exp = ctx->mk_ite(OVF2, pem2m1, biased_exp);
  smt_astt OVF = ctx->mk_or(OVF1, OVF2);

  assert(OVF2->sort->id == SMT_SORT_BOOL);
  assert(OVF->sort->id == SMT_SORT_BOOL);

  // ExpRnd
  smt_astt top_exp = mk_top_exp(ebits);
  smt_astt bot_exp = mk_bot_exp(ebits);

  smt_astt nil_1 = ctx->mk_smt_bv(BigInt(0), 1);

  smt_astt rm_is_to_zero = mk_is_rm(rm, ieee_floatt::ROUND_TO_ZERO);
  smt_astt rm_is_to_neg = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt rm_is_to_pos = mk_is_rm(rm, ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt rm_zero_or_neg = ctx->mk_or(rm_is_to_zero, rm_is_to_neg);
  smt_astt rm_zero_or_pos = ctx->mk_or(rm_is_to_zero, rm_is_to_pos);

  smt_astt zero1 = ctx->mk_smt_bv(BigInt(0), 1);
  smt_astt sgn_is_zero = ctx->mk_eq(sgn, zero1);

  smt_astt max_sig =
    ctx->mk_smt_bv(BigInt(power2m1(sbits - 1, false)), sbits - 1);
  smt_astt max_exp = ctx->mk_concat(
    ctx->mk_smt_bv(BigInt(power2m1(ebits - 1, false)), ebits - 1),
    ctx->mk_smt_bv(BigInt(0), 1));
  smt_astt inf_sig = ctx->mk_smt_bv(BigInt(0), sbits - 1);
  smt_astt inf_exp = top_exp;

  smt_astt max_inf_exp_neg = ctx->mk_ite(rm_zero_or_pos, max_exp, inf_exp);
  smt_astt max_inf_exp_pos = ctx->mk_ite(rm_zero_or_neg, max_exp, inf_exp);
  smt_astt ovfl_exp =
    ctx->mk_ite(sgn_is_zero, max_inf_exp_pos, max_inf_exp_neg);
  t_sig = ctx->mk_extract(sig, sbits - 1, sbits - 1);
  smt_astt n_d_check = ctx->mk_eq(t_sig, nil_1);
  smt_astt n_d_exp = ctx->mk_ite(n_d_check, bot_exp /* denormal */, biased_exp);
  exp = ctx->mk_ite(OVF, ovfl_exp, n_d_exp);

  smt_astt max_inf_sig_neg = ctx->mk_ite(rm_zero_or_pos, max_sig, inf_sig);
  smt_astt max_inf_sig_pos = ctx->mk_ite(rm_zero_or_neg, max_sig, inf_sig);
  smt_astt ovfl_sig =
    ctx->mk_ite(sgn_is_zero, max_inf_sig_pos, max_inf_sig_neg);
  smt_astt rest_sig = ctx->mk_extract(sig, sbits - 2, 0);
  sig = ctx->mk_ite(OVF, ovfl_sig, rest_sig);

  assert(sgn->sort->get_data_width() == 1);
  assert(sig->sort->get_data_width() == sbits - 1);
  assert(exp->sort->get_data_width() == ebits);

  result = mk_from_bv_to_fp(
    ctx->mk_concat(sgn, ctx->mk_concat(exp, sig)),
    mk_fpbv_sort(ebits, sbits - 1));
}

smt_astt fp_convt::mk_min_exp(std::size_t ebits)
{
  BigInt z = power2m1(ebits - 1, true) + 1;
  return ctx->mk_smt_bv(BigInt(z), ebits);
}

smt_astt fp_convt::mk_max_exp(std::size_t ebits)
{
  BigInt z = power2m1(ebits - 1, false);
  return ctx->mk_smt_bv(z, ebits);
}

smt_astt fp_convt::mk_top_exp(std::size_t sz)
{
  return ctx->mk_smt_bv(power2m1(sz, false), sz);
}

smt_astt fp_convt::mk_bot_exp(std::size_t sz)
{
  return ctx->mk_smt_bv(BigInt(0), sz);
}

smt_astt fp_convt::mk_rounding_decision(
  smt_astt &rm,
  smt_astt &sgn,
  smt_astt &last,
  smt_astt &round,
  smt_astt &sticky)
{
  smt_astt last_or_sticky = ctx->mk_bvor(last, sticky);
  smt_astt round_or_sticky = ctx->mk_bvor(round, sticky);

  smt_astt not_round = ctx->mk_bvnot(round);
  smt_astt not_lors = ctx->mk_bvnot(last_or_sticky);
  smt_astt not_rors = ctx->mk_bvnot(round_or_sticky);
  smt_astt not_sgn = ctx->mk_bvnot(sgn);

  smt_astt inc_teven = ctx->mk_bvnot(ctx->mk_bvor(not_round, not_lors));
  smt_astt inc_taway = round;
  smt_astt inc_pos = ctx->mk_bvnot(ctx->mk_bvor(sgn, not_rors));
  smt_astt inc_neg = ctx->mk_bvnot(ctx->mk_bvor(not_sgn, not_rors));

  smt_astt nil_1 = ctx->mk_smt_bv(BigInt(0), 1);

  smt_astt rm_is_to_neg = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt rm_is_to_pos = mk_is_rm(rm, ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt rm_is_away = mk_is_rm(rm, ieee_floatt::ROUND_TO_AWAY);
  smt_astt rm_is_even = mk_is_rm(rm, ieee_floatt::ROUND_TO_EVEN);

  smt_astt inc_c4 = ctx->mk_ite(rm_is_to_neg, inc_neg, nil_1);
  smt_astt inc_c3 = ctx->mk_ite(rm_is_to_pos, inc_pos, inc_c4);
  smt_astt inc_c2 = ctx->mk_ite(rm_is_away, inc_taway, inc_c3);
  return ctx->mk_ite(rm_is_even, inc_teven, inc_c2);
}

smt_astt fp_convt::mk_is_rm(smt_astt &rme, ieee_floatt::rounding_modet rm)
{
  smt_astt rm_num = ctx->mk_smt_bv(rm, 3);
  switch(rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
  case ieee_floatt::ROUND_TO_AWAY:
  case ieee_floatt::ROUND_TO_PLUS_INF:
  case ieee_floatt::ROUND_TO_MINUS_INF:
  case ieee_floatt::ROUND_TO_ZERO:
    return ctx->mk_eq(rme, rm_num);
  default:
    break;
  }

  std::cerr << "Unknown rounding mode\n";
  abort();
}

smt_astt fp_convt::mk_is_pos(smt_astt op)
{
  smt_astt sgn = extract_signbit(ctx, op);
  smt_astt zero = ctx->mk_smt_bv(BigInt(0), sgn->sort->get_data_width());
  return ctx->mk_eq(sgn, zero);
}

smt_astt fp_convt::mk_is_neg(smt_astt op)
{
  smt_astt sgn = extract_signbit(ctx, op);
  smt_astt one = ctx->mk_smt_bv(BigInt(1), sgn->sort->get_data_width());
  return ctx->mk_eq(sgn, one);
}

smt_astt fp_convt::mk_bias(smt_astt e)
{
  std::size_t ebits = e->sort->get_data_width();

  smt_astt bias = ctx->mk_smt_bv(BigInt(power2m1(ebits - 1, false)), ebits);
  return ctx->mk_bvadd(e, bias);
}

smt_astt fp_convt::mk_pzero(unsigned ew, unsigned sw)
{
  smt_astt bot_exp = mk_bot_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(0), 1),
      ctx->mk_concat(bot_exp, ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_nzero(unsigned ew, unsigned sw)
{
  smt_astt bot_exp = mk_bot_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(1), 1),
      ctx->mk_concat(bot_exp, ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_one(smt_astt sgn, unsigned ew, unsigned sw)
{
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      sgn,
      ctx->mk_concat(
        ctx->mk_smt_bv(BigInt(power2m1(ew - 1, false)), ew),
        ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_pinf(unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(0), 1),
      ctx->mk_concat(top_exp, ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_ninf(unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return mk_from_bv_to_fp(
    ctx->mk_concat(
      ctx->mk_smt_bv(BigInt(1), 1),
      ctx->mk_concat(top_exp, ctx->mk_smt_bv(BigInt(0), sw - 1))),
    mk_fpbv_sort(ew, sw - 1));
}

smt_astt fp_convt::mk_is_pzero(smt_astt op)
{
  return ctx->mk_and(
    ctx->fp_api->mk_smt_fpbv_is_zero(op), ctx->fp_api->mk_is_pos(op));
}

smt_astt fp_convt::mk_is_nzero(smt_astt op)
{
  return ctx->mk_and(
    ctx->fp_api->mk_smt_fpbv_is_zero(op), ctx->fp_api->mk_is_neg(op));
}

smt_astt fp_convt::mk_is_pinf(smt_astt op)
{
  return ctx->mk_and(
    ctx->fp_api->mk_smt_fpbv_is_inf(op), ctx->fp_api->mk_is_pos(op));
}

smt_astt fp_convt::mk_is_ninf(smt_astt op)
{
  return ctx->mk_and(
    ctx->fp_api->mk_smt_fpbv_is_inf(op), ctx->fp_api->mk_is_neg(op));
}

smt_astt fp_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  // Tricky, we need to change the type
  const_cast<smt_ast *>(op)->sort = to;
  return op;
}

smt_astt fp_convt::mk_from_fp_to_bv(smt_astt op)
{
  // Do nothing, it's already a bv
  return op;
}
