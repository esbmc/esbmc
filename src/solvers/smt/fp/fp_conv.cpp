#include <solvers/smt/smt_conv.h>

static smt_astt extract_exponent(smt_convt *ctx, smt_astt fp)
{
  std::size_t exp_top = fp->sort->get_data_width() - 2;
  std::size_t exp_bot = fp->sort->get_significand_width() - 2;
  std::size_t exp_width = exp_top - exp_bot;
  return ctx->mk_extract(
    fp, exp_top, exp_bot + 1, ctx->mk_bv_sort(SMT_SORT_UBV, exp_width));
}

static smt_astt extract_significand(smt_convt *ctx, smt_astt fp)
{
  return ctx->mk_extract(
    fp,
    fp->sort->get_significand_width() - 1,
    0,
    ctx->mk_bv_sort(SMT_SORT_UBV, fp->sort->get_significand_width()));
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
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)ew;
  (void)sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)sgn;
  (void)ew;
  (void)sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)rd;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)v1;
  (void)v2;
  (void)v3;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, smt_sortt to)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, smt_sortt to)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

ieee_floatt fp_convt::get_fpbv(smt_astt a)
{
  std::size_t width = a->sort->get_data_width();
  std::size_t swidth = a->sort->get_significand_width();

  ieee_floatt number(ieee_float_spect(swidth, width - swidth - 1));
  number.unpack(ctx->get_bv(a));
  return number;
}

smt_astt fp_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
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
  smt_astt ew_sw =
    ctx->mk_extract(op, op->sort->get_data_width() - 2, 0, zero->sort);

  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, ew_sw, zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  // Extract the sign bit
  smt_astt sign = ctx->mk_extract(
    op,
    op->sort->get_data_width() - 1,
    op->sort->get_data_width() - 1,
    zero->sort);

  // Compare with '0'
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, sign, zero);
}

smt_astt fp_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);

  // Extract the sign bit
  smt_astt sign = ctx->mk_extract(
    op,
    op->sort->get_data_width() - 1,
    op->sort->get_data_width() - 1,
    zero->sort);

  // Compare with '0'
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_NOTEQ, sign, zero);
}

smt_astt fp_convt::mk_smt_fpbv_abs(smt_astt op)
{
  smt_sortt no_sign_width =
    ctx->mk_bv_sort(SMT_SORT_UBV, op->sort->get_data_width() - 1);

  // Extract everything but the sign bit
  smt_astt ew_sw =
    ctx->mk_extract(op, op->sort->get_data_width() - 2, 0, no_sign_width);

  // Concat that with '0'
  smt_astt zero = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(0), 1);
  return ctx->mk_func_app(op->sort, SMT_FUNC_CONCAT, zero, ew_sw);
}
