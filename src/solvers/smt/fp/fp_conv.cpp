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
  return ctx->mk_extract(fp, fp->sort->get_significand_width() - 1, 0);
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

  // Create sign
  smt_astt sign = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  // All exponent bits are one
  smt_astt exp_all_ones =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), ew);

  // and significand is not zero
  smt_astt sig_all_zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), sw);

  // concat them all
  smt_sortt tmp_sort = ctx->mk_bv_sort(
    SMT_SORT_UBV,
    sign->sort->get_data_width() + exp_all_ones->sort->get_data_width());

  smt_astt sign_exp =
    ctx->mk_func_app(tmp_sort, SMT_FUNC_CONCAT, sign, exp_all_ones);

  smt_sortt s = ctx->mk_bv_fp_sort(ew, sw);
  return ctx->mk_func_app(s, SMT_FUNC_CONCAT, sign_exp, sig_all_zero);
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  // Create sign
  smt_astt sign = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(sgn), 1);

  // All exponent bits are one
  smt_astt exp_all_ones =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(ULONG_LONG_MAX), ew);

  // and all signficand bits are zero
  smt_astt sig_all_zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), sw);

  // concat them all
  smt_sortt tmp_sort = ctx->mk_bv_sort(
    SMT_SORT_UBV,
    sign->sort->get_data_width() + exp_all_ones->sort->get_data_width());

  smt_astt sign_exp =
    ctx->mk_func_app(tmp_sort, SMT_FUNC_CONCAT, sign, exp_all_ones);

  smt_sortt s = ctx->mk_bv_fp_sort(ew, sw);
  return ctx->mk_func_app(s, SMT_FUNC_CONCAT, sign_exp, sig_all_zero);
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  return ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(rm), 2);
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

  ieee_floatt number(ieee_float_spect(swidth, width - swidth - 1));
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

smt_astt fp_convt::mk_smt_fpbv_mul(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return ctx->convert_ast(
    float_bvt::mul(lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
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

  smt_astt comp = ctx->mk_func_app(
    ctx->boolean_sort, SMT_FUNC_ITE, signs_equal, ult, lhs_sign_eq_1);

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
  return ctx->mk_func_app(op->sort, SMT_FUNC_CONCAT, zero, ew_sw);
}

smt_astt fp_convt::mk_smt_fpbv_neg(smt_astt op)
{
  // We xor the sign bit with '1'
  smt_astt one = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1);
  smt_astt zeros =
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), op->sort->get_data_width() - 1);

  smt_astt one_zeros = ctx->mk_func_app(op->sort, SMT_FUNC_CONCAT, one, zeros);
  return ctx->mk_func_app(op->sort, SMT_FUNC_XOR, one_zeros, op);
}

fp_convt::unpacked_floatt fp_convt::unpack(smt_astt &src, bool normalize)
{
  unpacked_floatt res;

  unsigned sbits = src->sort->get_significand_width();
  unsigned ebits = src->sort->get_exponent_width();

  // Extract the sign bit
  res.sgn = extract_signbit(ctx, src);

  // Extract the exponent bits
  smt_astt exp = extract_exponent(ctx, src);

  // Extract the significand bits
  smt_astt sig = extract_significand(ctx, src);

  smt_astt is_normal = mk_smt_fpbv_is_normal(src);
  smt_astt normal_sig = ctx->mk_func_app(
    ctx->mk_bv_sort(SMT_SORT_UBV, sbits + 1),
    SMT_FUNC_CONCAT,
    ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(1), 1),
    sig);
  smt_astt normal_exp = mk_unbias(exp);

  smt_astt denormal_sig = ctx->mk_zero_ext(sig, 1);
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
    res.lz =
      ctx->mk_func_app(lz_d->sort, SMT_FUNC_ITE, norm_or_zero, zero_e, lz_d);

    smt_astt shift =
      ctx->mk_func_app(lz_d->sort, SMT_FUNC_ITE, is_sig_zero, zero_e, res.lz);
    if(ebits <= sbits)
    {
      smt_astt q = ctx->mk_zero_ext(shift, sbits - ebits);
      denormal_sig =
        ctx->mk_func_app(denormal_sig->sort, SMT_FUNC_SHL, denormal_sig, q);
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
      smt_astt sl = ctx->mk_func_app(
        sbits_s->sort, SMT_FUNC_ITE, is_sh_zero, short_shift, sbits_s);
      denormal_sig =
        ctx->mk_func_app(denormal_sig->sort, SMT_FUNC_SHL, denormal_sig, sl);
    }
  }
  else
    res.lz = zero_e;

  res.sig = ctx->mk_func_app(
    sig->sort, SMT_FUNC_ITE, is_normal, normal_sig, denormal_sig);

  res.exp = ctx->mk_func_app(
    sig->sort, SMT_FUNC_ITE, is_normal, normal_exp, denormal_exp);

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
  smt_astt n_leading = ctx->mk_func_app(leading->sort, SMT_FUNC_NOT, leading);
  smt_astt rest = ctx->mk_extract(e_plus_one, ebits - 2, 0);
  return ctx->mk_func_app(src->sort, SMT_FUNC_CONCAT, n_leading, rest);
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
    return ctx->mk_func_app(one_m->sort, SMT_FUNC_ITE, eq, one_m, nil_m);
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
    smt_astt sum = ctx->mk_func_app(lzL->sort, SMT_FUNC_ADD, h_m, lzL);
    return ctx->mk_func_app(lzH->sort, SMT_FUNC_ITE, H_is_zero, sum, lzH);
  }
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
