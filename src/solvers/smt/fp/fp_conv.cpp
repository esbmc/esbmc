#include <solvers/smt/smt_conv.h>
#include <solvers/smt/fp/float_bv.h>

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
  smt_sortt s = ctx->mk_bv_fp_sort(thereal.spec.e, thereal.spec.f);
  return ctx->mk_smt_bv(s, thereal.pack());
}

smt_sortt fp_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  return ctx->mk_bv_fp_sort(ew, sw);
}

smt_sortt fp_convt::mk_fpbv_rm_sort()
{
  return ctx->mk_bv_fp_rm_sort();
}

smt_astt fp_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  // TODO: we always create the same positive NaN:
  // 01111111100000000000000000000001
  smt_astt top_exp = mk_top_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1),
    ctx->mk_concat(top_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), sw - 1)));
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(sgn), 1),
    ctx->mk_concat(top_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw - 1)));
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  return ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(rm), 3);
}

smt_astt fp_convt::mk_smt_nearbyint_from_float(expr2tc from, expr2tc rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_sqrt(expr2tc rd, expr2tc rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)rd;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_fpbv_fma(expr2tc v1, expr2tc v2, expr2tc v3, expr2tc rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)v1;
  (void)v2;
  (void)v3;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_typecast_from_fpbv_to_ubv(expr2tc from, std::size_t width)
{
  return ctx->convert_ast(
    float_bvt::to_unsigned_integer(from, width, float_bvt::get_spec(from)));
}

smt_astt
fp_convt::mk_smt_typecast_from_fpbv_to_sbv(expr2tc from, std::size_t width)
{
  return ctx->convert_ast(
    float_bvt::to_signed_integer(from, width, float_bvt::get_spec(from)));
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  expr2tc from,
  type2tc to,
  expr2tc rm)
{
  return ctx->convert_ast(float_bvt::conversion(
    from,
    rm,
    float_bvt::get_spec(from),
    ieee_float_spect(to_floatbv_type(to))));
}

smt_astt
fp_convt::mk_smt_typecast_ubv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  return ctx->convert_ast(float_bvt::from_unsigned_integer(
    from, rm, ieee_float_spect(to_floatbv_type(to))));
}

smt_astt
fp_convt::mk_smt_typecast_sbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  return ctx->convert_ast(float_bvt::from_signed_integer(
    from, rm, ieee_float_spect(to_floatbv_type(to))));
}

ieee_floatt fp_convt::get_fpbv(smt_astt a)
{
  std::size_t width = a->sort->get_data_width();
  std::size_t swidth = a->sort->get_significand_width();

  ieee_floatt number(ieee_float_spect(swidth - 1, width - swidth));
  number.unpack(ctx->get_bv(a));
  return number;
}

smt_astt fp_convt::mk_smt_fpbv_add(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return ctx->convert_ast(float_bvt::add_sub(
    false, lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}

smt_astt fp_convt::mk_smt_fpbv_sub(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return ctx->convert_ast(float_bvt::add_sub(
    true, lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
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
  smt_astt x_is_pos = mk_smt_fpbv_is_positive(x);
  smt_astt y_is_nan = mk_smt_fpbv_is_nan(y);
  smt_astt y_is_zero = mk_smt_fpbv_is_zero(y);
  smt_astt y_is_pos = mk_smt_fpbv_is_positive(y);

  // (x is NaN) || (y is NaN) -> NaN
  smt_astt c1 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, x_is_nan, y_is_nan);
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
  smt_astt c6 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, x_is_zero, y_is_zero);
  smt_astt sign_xor =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_XOR, x_is_pos, y_is_pos);
  smt_astt v6 = ctx->mk_ite(sign_xor, nzero, pzero);

  // else comes the actual multiplication.
  unpacked_floatt a = unpack(x, true);
  unpacked_floatt b = unpack(y, true);

  smt_astt a_lz_ext = ctx->mk_zero_ext(a.lz, 2);
  smt_astt b_lz_ext = ctx->mk_zero_ext(b.lz, 2);

  smt_astt a_sig_ext = ctx->mk_zero_ext(a.sig, sbits);
  smt_astt b_sig_ext = ctx->mk_zero_ext(b.sig, sbits);

  smt_astt a_exp_ext = ctx->mk_zero_ext(a.exp, 2);
  smt_astt b_exp_ext = ctx->mk_zero_ext(b.exp, 2);

  unpacked_floatt res;
  res.sgn = ctx->mk_func_app(a.sgn->sort, SMT_FUNC_BVXOR, a.sgn, b.sgn);

  res.exp = ctx->mk_func_app(
    a_exp_ext->sort,
    SMT_FUNC_BVADD,
    ctx->mk_func_app(a_exp_ext->sort, SMT_FUNC_BVSUB, a_exp_ext, a_lz_ext),
    ctx->mk_func_app(b_exp_ext->sort, SMT_FUNC_BVSUB, b_exp_ext, b_lz_ext));

  smt_astt product =
    ctx->mk_func_app(a_sig_ext->sort, SMT_FUNC_BVMUL, a_sig_ext, b_sig_ext);

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
    rbits =
      ctx->mk_concat(l_p, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 4 - sbits));

  assert(rbits->sort->get_data_width() == 4);
  res.sig = ctx->mk_concat(h_p, rbits);

  smt_astt v7 = round(res, rm, ebits, sbits);

  // And finally, we tie them together.
  smt_astt result = ctx->mk_ite(c6, v6, v7);
  result = ctx->mk_ite(c5, v5, result);
  result = ctx->mk_ite(c4, v4, result);
  result = ctx->mk_ite(c3, v3, result);
  result = ctx->mk_ite(c2, v2, result);
  result = ctx->mk_ite(c1, v1, result);
  return result;
}

smt_astt fp_convt::mk_smt_fpbv_div(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return ctx->convert_ast(
    float_bvt::div(lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}

smt_astt fp_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  // Check if they are NaN
  smt_astt lhs_is_nan = mk_smt_fpbv_is_nan(lhs);
  smt_astt rhs_is_nan = mk_smt_fpbv_is_nan(rhs);
  smt_astt either_is_nan =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, lhs_is_nan, rhs_is_nan);

  // +0 and -0 should return true
  smt_astt lhs_is_zero = mk_smt_fpbv_is_zero(lhs);
  smt_astt rhs_is_zero = mk_smt_fpbv_is_zero(rhs);
  smt_astt both_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, lhs_is_zero, rhs_is_zero);

  // Otherwise compare them bitwise
  smt_astt are_equal =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, lhs, rhs);

  // They are equal if they are either +0 and -0 (and vice-versa) or bitwise
  // equal and neither is NaN
  smt_astt either_zero_or_equal =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, both_zero, are_equal);

  smt_astt not_nan =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOT, either_is_nan);

  return ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_AND, either_zero_or_equal, not_nan);
}

smt_astt fp_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  // (a > b) iff (b < a)
  return mk_smt_fpbv_lt(rhs, lhs);
}

smt_astt fp_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  // Check if they are NaN
  smt_astt lhs_is_nan = mk_smt_fpbv_is_nan(lhs);
  smt_astt rhs_is_nan = mk_smt_fpbv_is_nan(rhs);
  smt_astt either_is_nan =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, lhs_is_nan, rhs_is_nan);
  smt_astt not_nan =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOT, either_is_nan);

  // +0 and -0 should return false
  smt_astt lhs_is_zero = mk_smt_fpbv_is_zero(lhs);
  smt_astt rhs_is_zero = mk_smt_fpbv_is_zero(rhs);
  smt_astt both_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, lhs_is_zero, rhs_is_zero);
  smt_astt not_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOT, both_zero);

  // TODO: we do an unsigned comparison, but due to the bias, we should safe
  // to do a signed comparison.

  // Extract the exponents, significands and signs
  smt_astt lhs_exp_sig = extract_exp_sig(ctx, lhs);
  smt_astt lhs_sign = extract_signbit(ctx, lhs);

  smt_astt rhs_exp_sig = extract_exp_sig(ctx, rhs);
  smt_astt rhs_sign = extract_signbit(ctx, lhs);

  // Compare signs
  smt_astt signs_equal =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, lhs_sign, rhs_sign);

  // Compare the exp_sign
  smt_astt ult = ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_BVULT, lhs_exp_sig, rhs_exp_sig);

  // If the signs are equal, return x < y, otherwise return the sign of y
  smt_astt lhs_sign_eq_1 = ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_EQ,
    lhs_sign,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1));

  smt_astt comp = ctx->mk_ite(signs_equal, ult, lhs_sign_eq_1);

  smt_astt not_zeros_not_nan =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, not_zero, not_nan);

  return ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_AND, not_zeros_not_nan, comp);
}

smt_astt fp_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  // This is !FPLT
  smt_astt a = mk_smt_fpbv_lt(lhs, rhs);
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOT, a);
}

smt_astt fp_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  smt_astt lt = mk_smt_fpbv_lt(lhs, rhs);
  smt_astt eq = mk_smt_fpbv_eq(lhs, rhs);
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, lt, eq);
}

smt_astt fp_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  // Extract the exponent bits
  smt_astt exp = extract_exponent(ctx, op);

  // Extract the significand bits
  smt_astt sig = extract_significand(ctx, op);

  // A fp is NaN if all bits in the exponent are ones
  smt_astt all_ones = ctx->mk_smt_bv(
    SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), exp->sort->get_data_width());

  smt_astt exp_all_ones =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, exp, all_ones);

  // and all bits in the significand are not zero
  smt_astt zero =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sig->sort->get_data_width());

  smt_astt sig_all_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOTEQ, sig, zero);

  return ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_AND, exp_all_ones, sig_all_zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  // Extract the exponent bits
  smt_astt exp = extract_exponent(ctx, op);

  // Extract the significand bits
  smt_astt sig = extract_significand(ctx, op);

  // A fp is inf if all bits in the exponent are ones
  smt_astt all_ones = ctx->mk_smt_bv(
    SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), exp->sort->get_data_width());

  smt_astt exp_all_ones =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, exp, all_ones);

  // and the significand is zero
  smt_astt zero =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sig->sort->get_data_width());

  smt_astt sig_all_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, sig, zero);

  return ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_AND, exp_all_ones, sig_all_zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  // Extract the exponent bits
  smt_astt exp = extract_exponent(ctx, op);

  // Extract the significand bits
  smt_astt sig = extract_significand(ctx, op);

  // A fp is normal if the exponent is not zero
  smt_astt zero =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), exp->sort->get_data_width());

  smt_astt exp_not_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOTEQ, exp, zero);

  // and the all bits in the significand are not one
  smt_astt all_ones = ctx->mk_smt_bv(
    SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), sig->sort->get_data_width());

  smt_astt sig_not_all_ones =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOTEQ, sig, all_ones);

  return ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_AND, exp_not_zero, sig_not_all_ones);
}

smt_astt fp_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  // Both -0 and 0 should return true

  // Compare with '0'
  smt_astt zero =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), op->sort->get_data_width() - 1);

  // Extract everything but the sign bit
  smt_astt ew_sw = extract_exp_sig(ctx, op);

  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, ew_sw, zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  // Extract the sign bit
  smt_astt sign = extract_signbit(ctx, op);

  // Compare with '0'
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOTEQ, sign, zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  // Extract the sign bit
  smt_astt sign = extract_signbit(ctx, op);

  // Compare with '0'
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, sign, zero);
}

smt_astt fp_convt::mk_smt_fpbv_abs(smt_astt op)
{
  // Extract everything but the sign bit
  smt_astt ew_sw = extract_exp_sig(ctx, op);

  // Concat that with '0'
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);
  return ctx->mk_concat(zero, ew_sw);
}

smt_astt fp_convt::mk_smt_fpbv_neg(smt_astt op)
{
  // We xor the sign bit with '1'
  smt_astt one = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1);
  smt_astt zeros =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), op->sort->get_data_width() - 1);

  smt_astt one_zeros = ctx->mk_concat(one, zeros);
  return ctx->mk_func_app(op->sort, SMT_FUNC_XOR, one_zeros, op);
}

fp_convt::unpacked_floatt fp_convt::unpack(smt_astt &src, bool normalize)
{
  unpacked_floatt res;

  unsigned sbits = src->sort->get_significand_width();
  unsigned ebits = src->sort->get_exponent_width();

  // Extract parts
  res.sgn = extract_signbit(ctx, src);
  res.exp = extract_exponent(ctx, src);
  res.sig = extract_significand(ctx, src);

  assert(res.sgn->sort->get_data_width() == 1);
  assert(res.exp->sort->get_data_width() == ebits);
  assert(res.sig->sort->get_data_width() == sbits - 1);

  smt_astt is_normal = mk_smt_fpbv_is_normal(src);
  smt_astt normal_sig =
    ctx->mk_concat(ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1), res.sig);
  smt_astt normal_exp = mk_unbias(res.exp);

  smt_astt denormal_sig = ctx->mk_zero_ext(res.sig, 1);
  smt_astt denormal_exp = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits);
  denormal_exp = mk_unbias(denormal_exp);

  smt_astt zero_e = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), ebits);
  if(normalize)
  {
    smt_astt zero_s = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sbits);
    smt_astt is_sig_zero =
      ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, zero_s, denormal_sig);

    smt_astt lz_d = mk_leading_zeros(denormal_sig, ebits);
    smt_astt norm_or_zero =
      ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, is_normal, is_sig_zero);
    res.lz = ctx->mk_ite(norm_or_zero, zero_e, lz_d);

    smt_astt shift = ctx->mk_ite(is_sig_zero, zero_e, res.lz);
    assert(shift->sort->get_data_width() == ebits);
    if(ebits <= sbits)
    {
      smt_astt q = ctx->mk_zero_ext(shift, sbits - ebits);
      denormal_sig =
        ctx->mk_func_app(denormal_sig->sort, SMT_FUNC_BVSHL, denormal_sig, q);
    }
    else
    {
      // the maximum shift is `sbits', because after that the mantissa
      // would be zero anyways. So we can safely cut the shift variable down,
      // as long as we check the higher bits.
      smt_astt zero_ems =
        ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), ebits - sbits);
      smt_astt sbits_s = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(sbits), sbits);
      smt_astt sh = ctx->mk_extract(shift, ebits - 1, sbits);
      smt_astt is_sh_zero =
        ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, zero_ems, sh);
      smt_astt short_shift = ctx->mk_extract(shift, sbits - 1, 0);
      smt_astt sl = ctx->mk_ite(is_sh_zero, short_shift, sbits_s);
      denormal_sig =
        ctx->mk_func_app(denormal_sig->sort, SMT_FUNC_BVSHL, denormal_sig, sl);
    }
  }
  else
    res.lz = zero_e;

  res.sig = ctx->mk_ite(is_normal, normal_sig, denormal_sig);
  res.exp = ctx->mk_ite(is_normal, normal_exp, denormal_exp);

  assert(res.sgn->sort->get_data_width() == 1);
  assert(res.sig->sort->get_data_width() == sbits);
  assert(res.exp->sort->get_data_width() == ebits);

  return res;
}

smt_astt fp_convt::mk_unbias(smt_astt &src)
{
  unsigned ebits = src->sort->get_data_width();

  smt_astt e_plus_one = ctx->mk_func_app(
    src->sort,
    SMT_FUNC_BVADD,
    src,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits));

  smt_astt leading = ctx->mk_extract(e_plus_one, ebits - 1, ebits - 1);
  smt_astt n_leading = ctx->mk_func_app(leading->sort, SMT_FUNC_BVNOT, leading);
  smt_astt rest = ctx->mk_extract(e_plus_one, ebits - 2, 0);
  return ctx->mk_concat(n_leading, rest);
}

smt_astt fp_convt::mk_leading_zeros(smt_astt &src, std::size_t max_bits)
{
  std::size_t bv_sz = src->sort->get_data_width();

  if(bv_sz == 0)
    return ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), max_bits);
  else if(bv_sz == 1)
  {
    smt_astt nil_1 = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);
    smt_astt one_m = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), max_bits);
    smt_astt nil_m = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), max_bits);

    smt_astt eq = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, src, nil_1);
    return ctx->mk_ite(eq, one_m, nil_m);
  }
  else
  {
    smt_astt H = ctx->mk_extract(src, bv_sz - 1, bv_sz / 2);
    smt_astt L = ctx->mk_extract(src, bv_sz / 2 - 1, 0);

    unsigned H_size = H->sort->get_data_width();

    smt_astt lzH = mk_leading_zeros(H, max_bits); /* recursive! */
    smt_astt lzL = mk_leading_zeros(L, max_bits);

    smt_astt nil_h = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), H_size);
    smt_astt H_is_zero =
      ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, H, nil_h);

    smt_astt h_m = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(H_size), max_bits);
    smt_astt sum = ctx->mk_func_app(lzL->sort, SMT_FUNC_BVADD, h_m, lzL);
    return ctx->mk_ite(H_is_zero, sum, lzH);
  }
}

smt_astt fp_convt::round(
  unpacked_floatt &src,
  smt_astt &rm,
  unsigned ebits,
  unsigned sbits)
{
  // Assumptions: sig is of the form f[-1:0] . f[1:sbits-1] [guard,round,sticky],
  // i.e., it has 2 + (sbits-1) + 3 = sbits + 4 bits, where the first one is in sgn.
  // Furthermore, note that sig is an unsigned bit-vector, while exp is signed.

  smt_astt e_min = mk_min_exp(ebits);
  smt_astt e_max = mk_max_exp(ebits);

  smt_astt one_1 = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1);
  smt_astt h_exp = ctx->mk_extract(src.exp, ebits + 1, ebits + 1);
  smt_astt sh_exp = ctx->mk_extract(src.exp, ebits, ebits);
  smt_astt th_exp = ctx->mk_extract(src.exp, ebits - 1, ebits - 1);
  smt_astt e3 = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, h_exp, one_1);
  smt_astt e2 = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, sh_exp, one_1);
  smt_astt e1 = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, th_exp, one_1);
  smt_astt e21 = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, e2, e1);
  smt_astt ne3 = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOT, e3);
  smt_astt e_top_three =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, ne3, e21);

  smt_astt ext_emax = ctx->mk_zero_ext(e_max, 2);
  smt_astt t_sig = ctx->mk_extract(src.sig, sbits + 3, sbits + 3);
  smt_astt e_eq_emax =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, ext_emax, src.exp);
  smt_astt sigm1 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, t_sig, one_1);
  smt_astt e_eq_emax_and_sigm1 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, e_eq_emax, sigm1);
  smt_astt OVF1 = ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_OR, e_top_three, e_eq_emax_and_sigm1);

  // CMW: is this always large enough?
  smt_astt lz = mk_leading_zeros(src.sig, ebits + 2);

  smt_astt t = ctx->mk_func_app(
    src.exp->sort,
    SMT_FUNC_BVADD,
    src.exp,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits + 2));
  t = ctx->mk_func_app(src.exp->sort, SMT_FUNC_BVSUB, t, lz);
  t = ctx->mk_func_app(
    src.exp->sort, SMT_FUNC_BVSUB, t, ctx->mk_sign_ext(e_min, 2));
  smt_astt TINY = ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_BVSLTE,
    t,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), ebits + 2));

  smt_astt beta = ctx->mk_func_app(
    src.exp->sort,
    SMT_FUNC_BVADD,
    ctx->mk_func_app(src.exp->sort, SMT_FUNC_BVSUB, src.exp, lz),
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits + 2));

  smt_astt sigma_add = ctx->mk_func_app(
    src.exp->sort, SMT_FUNC_BVSUB, src.exp, ctx->mk_sign_ext(e_min, 2));
  sigma_add = ctx->mk_func_app(
    sigma_add->sort,
    SMT_FUNC_BVADD,
    sigma_add,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits + 2));
  smt_astt sigma = ctx->mk_ite(TINY, sigma_add, lz);

  // Normalization shift
  std::size_t sig_size = src.sig->sort->get_data_width();
  std::size_t sigma_size = ebits + 2;

  smt_astt sigma_neg = ctx->mk_func_app(sigma->sort, SMT_FUNC_BVNEG, sigma);
  smt_astt sigma_cap = ctx->mk_smt_bv(SMT_SORT_UBV, sbits + 2, sigma_size);
  smt_astt sigma_le_cap =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_BVULTE, sigma_neg, sigma_cap);
  smt_astt sigma_neg_capped = ctx->mk_ite(sigma_le_cap, sigma_neg, sigma_cap);
  smt_astt sigma_lt_zero = ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_BVSLTE,
    sigma,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), sigma_size));

  smt_astt sig_ext =
    ctx->mk_concat(src.sig, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sig_size));
  smt_astt rs_sig = ctx->mk_func_app(
    sig_ext->sort,
    SMT_FUNC_BVLSHR,
    sig_ext,
    ctx->mk_zero_ext(sigma_neg_capped, 2 * sig_size - sigma_size));
  smt_astt ls_sig = ctx->mk_func_app(
    sig_ext->sort,
    SMT_FUNC_BVSHL,
    sig_ext,
    ctx->mk_zero_ext(sigma, 2 * sig_size - sigma_size));
  smt_astt big_sh_sig = ctx->mk_ite(sigma_lt_zero, rs_sig, ls_sig);

  std::size_t sig_extract_low_bit = (2 * sig_size - 1) - (sbits + 2) + 1;
  src.sig = ctx->mk_extract(big_sh_sig, 2 * sig_size - 1, sig_extract_low_bit);

  smt_astt sticky =
    ctx->mk_bvredor(ctx->mk_extract(big_sh_sig, sig_extract_low_bit - 1, 0));

  // put the sticky bit into the significand.
  smt_astt ext_sticky = ctx->mk_zero_ext(sticky, sbits + 1);
  src.sig = ctx->mk_func_app(src.sig->sort, SMT_FUNC_BVOR, src.sig, ext_sticky);

  smt_astt ext_emin = ctx->mk_zero_ext(e_min, 2);
  src.exp = ctx->mk_ite(TINY, ext_emin, beta);

  // Significand rounding
  sticky = ctx->mk_extract(src.sig, 0, 0); // new sticky bit!
  smt_astt round = ctx->mk_extract(src.sig, 1, 1);
  smt_astt last = ctx->mk_extract(src.sig, 2, 2);

  src.sig = ctx->mk_extract(src.sig, sbits + 1, 2);

  smt_astt inc = mk_rounding_decision(rm, src.sgn, last, round, sticky);

  src.sig = ctx->mk_func_app(
    ctx->mk_bv_sort(SMT_SORT_UBV, src.sig->sort->get_data_width() + 1),
    SMT_FUNC_BVADD,
    ctx->mk_zero_ext(src.sig, 1),
    ctx->mk_zero_ext(inc, sbits));

  t_sig = ctx->mk_extract(src.sig, sbits, sbits);
  smt_astt SIGovf =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, t_sig, one_1);

  smt_astt hallbut1_sig = ctx->mk_extract(src.sig, sbits, 1);
  smt_astt lallbut1_sig = ctx->mk_extract(src.sig, sbits - 1, 0);
  src.sig = ctx->mk_ite(SIGovf, hallbut1_sig, lallbut1_sig);

  smt_astt exp_p1 = ctx->mk_func_app(
    src.exp->sort,
    SMT_FUNC_BVADD,
    src.exp,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), ebits + 2));
  src.exp = ctx->mk_ite(SIGovf, exp_p1, src.exp);

  // Exponent adjustment and rounding
  smt_astt biased_exp = mk_bias(ctx->mk_extract(src.exp, ebits - 1, 0));

  // AdjustExp
  smt_astt exp_redand = ctx->mk_bvredand(biased_exp);
  smt_astt preOVF2 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, exp_redand, one_1);
  smt_astt OVF2 =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_AND, SIGovf, preOVF2);
  smt_astt pem2m1 =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(power2m1(ebits - 2, false)), ebits);
  biased_exp = ctx->mk_ite(OVF2, pem2m1, biased_exp);
  smt_astt OVF = ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_OR, OVF1, OVF2);

  // ExpRnd
  smt_astt top_exp = mk_top_exp(ebits);
  smt_astt bot_exp = mk_bot_exp(ebits);

  smt_astt nil_1 = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  smt_astt rm_is_to_zero = mk_is_rm(rm, ieee_floatt::ROUND_TO_ZERO);
  smt_astt rm_is_to_neg = mk_is_rm(rm, ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt rm_is_to_pos = mk_is_rm(rm, ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt rm_zero_or_neg = ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_OR, rm_is_to_zero, rm_is_to_neg);
  smt_astt rm_zero_or_pos = ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_OR, rm_is_to_zero, rm_is_to_pos);

  smt_astt zero1 = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);
  smt_astt sgn_is_zero =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, src.sgn, zero1);

  smt_astt max_sig =
    ctx->mk_smt_bv(SMT_SORT_UBV, power2m1(sbits - 1, false), sbits - 1);
  smt_astt max_exp = ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, power2m1(ebits - 1, false), ebits - 1),
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1));
  smt_astt inf_sig = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sbits - 1);
  smt_astt inf_exp = top_exp;

  smt_astt max_inf_exp_neg = ctx->mk_ite(rm_zero_or_pos, max_exp, inf_exp);
  smt_astt max_inf_exp_pos = ctx->mk_ite(rm_zero_or_neg, max_exp, inf_exp);
  smt_astt ovfl_exp =
    ctx->mk_ite(sgn_is_zero, max_inf_exp_pos, max_inf_exp_neg);
  t_sig = ctx->mk_extract(src.sig, sbits - 1, sbits - 1);
  smt_astt n_d_check =
    ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, t_sig, nil_1);
  smt_astt n_d_exp = ctx->mk_ite(n_d_check, bot_exp /* denormal */, biased_exp);
  src.exp = ctx->mk_ite(OVF, ovfl_exp, n_d_exp);

  smt_astt max_inf_sig_neg = ctx->mk_ite(rm_zero_or_pos, max_sig, inf_sig);
  smt_astt max_inf_sig_pos = ctx->mk_ite(rm_zero_or_neg, max_sig, inf_sig);
  smt_astt ovfl_sig =
    ctx->mk_ite(sgn_is_zero, max_inf_sig_pos, max_inf_sig_neg);
  smt_astt rest_sig = ctx->mk_extract(src.sig, sbits - 2, 0);
  src.sig = ctx->mk_ite(OVF, ovfl_sig, rest_sig);

  smt_astt res_exp_sig = ctx->mk_concat(src.exp, src.sig);

  return ctx->mk_concat(src.sgn, res_exp_sig);
}

smt_astt fp_convt::mk_min_exp(std::size_t ebits)
{
  BigInt z = power2m1(ebits - 1, true) + 1;
  return ctx->mk_smt_bv(SMT_SORT_SBV, z, ebits);
}

smt_astt fp_convt::mk_max_exp(std::size_t ebits)
{
  BigInt z = power2m1(ebits - 1, false);
  return ctx->mk_smt_bv(SMT_SORT_UBV, z, ebits);
}

smt_astt fp_convt::mk_top_exp(std::size_t sz)
{
  return ctx->mk_smt_bv(SMT_SORT_UBV, power2m1(sz, false), sz);
}

smt_astt fp_convt::mk_bot_exp(std::size_t sz)
{
  return ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sz);
}

smt_astt fp_convt::mk_rounding_decision(
  smt_astt &rm,
  smt_astt &sgn,
  smt_astt &last,
  smt_astt &round,
  smt_astt &sticky)
{
  smt_astt last_or_sticky =
    ctx->mk_func_app(last->sort, SMT_FUNC_BVOR, last, sticky);
  smt_astt round_or_sticky =
    ctx->mk_func_app(round->sort, SMT_FUNC_BVOR, round, sticky);

  smt_astt not_round = ctx->mk_func_app(round->sort, SMT_FUNC_BVNOT, round);
  smt_astt not_lors =
    ctx->mk_func_app(last_or_sticky->sort, SMT_FUNC_BVNOT, last_or_sticky);
  smt_astt not_rors =
    ctx->mk_func_app(round_or_sticky->sort, SMT_FUNC_BVNOT, round_or_sticky);
  smt_astt not_sgn = ctx->mk_func_app(sgn->sort, SMT_FUNC_BVNOT, sgn);

  smt_astt inc_teven = ctx->mk_func_app(
    not_round->sort,
    SMT_FUNC_BVNOT,
    ctx->mk_func_app(last->sort, SMT_FUNC_BVOR, not_round, not_lors));
  smt_astt inc_taway = round;
  smt_astt inc_pos = ctx->mk_func_app(
    sgn->sort,
    SMT_FUNC_BVNOT,
    ctx->mk_func_app(sgn->sort, SMT_FUNC_BVOR, sgn, not_rors));
  smt_astt inc_neg = ctx->mk_func_app(
    not_sgn->sort,
    SMT_FUNC_BVNOT,
    ctx->mk_func_app(not_sgn->sort, SMT_FUNC_BVOR, not_sgn, not_rors));

  smt_astt nil_1 = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

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
  smt_astt rm_num = ctx->mk_smt_bv(SMT_SORT_UBV, rm, 3);
  switch(rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
  case ieee_floatt::ROUND_TO_AWAY:
  case ieee_floatt::ROUND_TO_PLUS_INF:
  case ieee_floatt::ROUND_TO_MINUS_INF:
  case ieee_floatt::ROUND_TO_ZERO:
    return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, rme, rm_num);
  default:
    break;
  }

  std::cerr << "Unknown rounding mode\n";
  abort();
}

smt_astt fp_convt::mk_bias(smt_astt e)
{
  std::size_t ebits = e->sort->get_data_width();

  smt_astt bias =
    ctx->mk_smt_bv(SMT_SORT_SBV, power2m1(ebits - 1, false), ebits);
  return ctx->mk_func_app(e->sort, SMT_FUNC_BVADD, e, bias);
}

smt_astt fp_convt::mk_pzero(unsigned ew, unsigned sw)
{
  smt_astt bot_exp = mk_bot_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1),
    ctx->mk_concat(bot_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw - 1)));
}

smt_astt fp_convt::mk_nzero(unsigned ew, unsigned sw)
{
  smt_astt bot_exp = mk_bot_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1),
    ctx->mk_concat(bot_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw - 1)));
}

smt_astt fp_convt::mk_pinf(unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1),
    ctx->mk_concat(top_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw - 1)));
}

smt_astt fp_convt::mk_ninf(unsigned ew, unsigned sw)
{
  smt_astt top_exp = mk_top_exp(ew);
  return ctx->mk_concat(
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1),
    ctx->mk_concat(top_exp, ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw - 1)));
}

smt_astt fp_convt::mk_is_pzero(smt_astt op)
{
  return ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_AND,
    ctx->fp_api->mk_smt_fpbv_is_zero(op),
    ctx->fp_api->mk_smt_fpbv_is_positive(op));
}

smt_astt fp_convt::mk_is_nzero(smt_astt op)
{
  return ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_AND,
    ctx->fp_api->mk_smt_fpbv_is_zero(op),
    ctx->fp_api->mk_smt_fpbv_is_negative(op));
}

smt_astt fp_convt::mk_is_pinf(smt_astt op)
{
  return ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_AND,
    ctx->fp_api->mk_smt_fpbv_is_inf(op),
    ctx->fp_api->mk_smt_fpbv_is_positive(op));
}

smt_astt fp_convt::mk_is_ninf(smt_astt op)
{
  return ctx->mk_func_app(
    ctx->boolean_sort,
    SMT_FUNC_AND,
    ctx->fp_api->mk_smt_fpbv_is_inf(op),
    ctx->fp_api->mk_smt_fpbv_is_negative(op));
}
