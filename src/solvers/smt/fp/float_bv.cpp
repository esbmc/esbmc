/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <algorithm>
#include <cassert>
#include <solvers/smt/fp/float_bv.h>
#include <util/arith_tools.h>
#include <util/std_expr.h>

ieee_float_spect float_bvt::get_spec(const expr2tc &expr)
{
  const floatbv_type2tc &type = to_floatbv_type(expr->type);
  return ieee_float_spect(type);
}

expr2tc
float_bvt::exponent_all_ones(const expr2tc &src, const ieee_float_spect &spec)
{
  expr2tc exponent = get_exponent(src, spec);
  expr2tc all_ones = from_integer(BigInt(ULONG_LONG_MAX), exponent->type);
  return equality2tc(exponent, all_ones);
}

expr2tc float_bvt::is_zero(const expr2tc &src, const ieee_float_spect &spec)
{
  return and2tc(exponent_all_zeros(src, spec), fraction_all_zeros(src, spec));
}
expr2tc
float_bvt::exponent_all_zeros(const expr2tc &src, const ieee_float_spect &spec)
{
  expr2tc exponent = get_exponent(src, spec);
  expr2tc all_zeros = gen_zero(exponent->type);
  return equality2tc(exponent, all_zeros);
}

expr2tc
float_bvt::fraction_all_zeros(const expr2tc &src, const ieee_float_spect &spec)
{
  // does not include hidden bit
  expr2tc fraction = get_fraction(src, spec);
  expr2tc all_zeros = gen_zero(fraction->type);
  return equality2tc(fraction, all_zeros);
}

void float_bvt::rounding_mode_bitst::get(const expr2tc &rm)
{
  expr2tc round_to_even_const =
    from_integer(ieee_floatt::ROUND_TO_EVEN, rm->type);
  expr2tc round_to_plus_inf_const =
    from_integer(ieee_floatt::ROUND_TO_PLUS_INF, rm->type);
  expr2tc round_to_minus_inf_const =
    from_integer(ieee_floatt::ROUND_TO_MINUS_INF, rm->type);
  expr2tc round_to_zero_const =
    from_integer(ieee_floatt::ROUND_TO_ZERO, rm->type);

  round_to_even = equality2tc(rm, round_to_even_const);
  round_to_plus_inf = equality2tc(rm, round_to_plus_inf_const);
  round_to_minus_inf = equality2tc(rm, round_to_minus_inf_const);
  round_to_zero = equality2tc(rm, round_to_zero_const);
}

expr2tc float_bvt::sign_bit(const expr2tc &op)
{
  std::size_t width = op->type->get_width();
  return extract2tc(type_pool.get_uint(1), op, width - 1, width - 1);
}

expr2tc float_bvt::from_signed_integer(
  const expr2tc &src,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  std::size_t src_width = src->type->get_width();

  unbiased_floatt result;

  // we need to adjust for negative integers
  result.sign = sign_bit(src);

  result.fraction =
    typecast2tc(type_pool.get_uint(src_width), abs2tc(src->type, src));

  // build an exponent (unbiased) -- this is signed!
  result.exponent = from_integer(
    src_width - 1,
    type_pool.get_int(address_bits(src_width - 1).to_long() + 1));

  return rounder(result, rm, spec);
}

expr2tc float_bvt::from_unsigned_integer(
  const expr2tc &src,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  unbiased_floatt result;

  result.fraction = src;

  std::size_t src_width = src->type->get_width();

  // build an exponent (unbiased) -- this is signed!
  result.exponent = from_integer(
    src_width - 1,
    type_pool.get_int(address_bits(src_width - 1).to_long() + 1));

  result.sign = gen_false_expr();

  return rounder(result, rm, spec);
}

expr2tc float_bvt::to_signed_integer(
  const expr2tc &src,
  std::size_t dest_width,
  const ieee_float_spect &spec)
{
  return to_integer(src, dest_width, true, spec);
}

expr2tc float_bvt::to_unsigned_integer(
  const expr2tc &src,
  std::size_t dest_width,
  const ieee_float_spect &spec)
{
  return to_integer(src, dest_width, false, spec);
}

expr2tc float_bvt::to_integer(
  const expr2tc &src,
  std::size_t dest_width,
  bool is_signed,
  const ieee_float_spect &spec)
{
  const unbiased_floatt unpacked = unpack(src, spec);

  // Right now hard-wired to round-to-zero, which is
  // the usual case in ANSI-C.

  // if the exponent is positive, shift right
  expr2tc offset = from_integer(spec.f, type_pool.get_int(spec.e));
  expr2tc distance = sub2tc(offset->type, offset, unpacked.exponent);
  expr2tc shift_result =
    lshr2tc(unpacked.fraction->type, unpacked.fraction, distance);

  // if the exponent is negative, we have zero anyways
  expr2tc result = shift_result;
  expr2tc exponent_sign = sign_bit(unpacked.exponent);

  result = if2tc(result->type, exponent_sign, gen_zero(result->type), result);

  // chop out the right number of bits from the result
  type2tc result_type =
    is_signed ? type_pool.get_int(dest_width) : type_pool.get_uint(dest_width);

  result = typecast2tc(result_type, result);

  // if signed, apply sign.
  if(is_signed)
  {
    result =
      if2tc(result->type, unpacked.sign, neg2tc(result->type, result), result);
  }
  else
  {
    // It's unclear what the behaviour for negative floats
    // to integer shall be.
  }

  return result;
}

expr2tc float_bvt::conversion(
  const expr2tc &src,
  const expr2tc &rm,
  const ieee_float_spect &src_spec,
  const ieee_float_spect &dest_spec)
{
  // Catch the special case in which we extend,
  // e.g. single to double.
  // In this case, rounding can be avoided,
  // but a denormal number may be come normal.
  // Be careful to exclude the difficult case
  // when denormalised numbers in the old format
  // can be converted to denormalised numbers in the
  // new format.  Note that this is rare and will only
  // happen with very non-standard formats.

  int sourceSmallestNormalExponent = -((1 << (src_spec.e - 1)) - 1);
  int sourceSmallestDenormalExponent =
    sourceSmallestNormalExponent - src_spec.f;

  // Using the fact that f doesn't include the hidden bit

  int destSmallestNormalExponent = -((1 << (dest_spec.e - 1)) - 1);

  if(
    dest_spec.e >= src_spec.e && dest_spec.f >= src_spec.f &&
    !(sourceSmallestDenormalExponent < destSmallestNormalExponent))
  {
    unbiased_floatt unpacked_src = unpack(src, src_spec);
    unbiased_floatt result;

    // the fraction gets zero-padded
    std::size_t padding = dest_spec.f - src_spec.f;
    result.fraction = concat2tc(
      type_pool.get_uint(dest_spec.f + 1),
      unpacked_src.fraction,
      gen_zero(type_pool.get_uint(padding)));

    // the exponent gets sign-extended
    assert(is_signedbv_type(unpacked_src.exponent));
    result.exponent =
      typecast2tc(type_pool.get_int(dest_spec.e), unpacked_src.exponent);

    // if the number was denormal and is normal in the new format,
    // normalise it!
    if(dest_spec.e > src_spec.e)
      normalization_shift(result.fraction, result.exponent);

    // the flags get copied
    result.sign = unpacked_src.sign;
    result.NaN = unpacked_src.NaN;
    result.infinity = unpacked_src.infinity;

    // no rounding needed!
    return pack(bias(result, dest_spec), dest_spec);
  }
  else
  {
    // we actually need to round
    unbiased_floatt result = unpack(src, src_spec);
    return rounder(result, rm, dest_spec);
  }
}

expr2tc float_bvt::isnormal(const expr2tc &src, const ieee_float_spect &spec)
{
  return and2tc(
    not2tc(exponent_all_zeros(src, spec)),
    not2tc(exponent_all_ones(src, spec)));
}

/// Subtracts the exponents
expr2tc float_bvt::subtract_exponents(
  const unbiased_floatt &src1,
  const unbiased_floatt &src2)
{
  // extend both by one bit
  std::size_t old_width1 = src1.exponent->type->get_width();
  std::size_t old_width2 = src2.exponent->type->get_width();
  assert(old_width1 == old_width2);

  expr2tc extended_exponent1 =
    typecast2tc(type_pool.get_int(old_width1 + 1), src1.exponent);
  expr2tc extended_exponent2 =
    typecast2tc(type_pool.get_int(old_width2 + 1), src2.exponent);

  assert(extended_exponent1->type == extended_exponent2->type);

  // compute shift distance (here is the subtraction)
  return sub2tc(
    extended_exponent1->type, extended_exponent1, extended_exponent2);
}

expr2tc float_bvt::add_sub(
  bool subtract,
  const expr2tc &op0,
  const expr2tc &op1,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  unbiased_floatt unpacked1 = unpack(op0, spec);
  unbiased_floatt unpacked2 = unpack(op1, spec);

  // subtract?
  if(subtract)
    unpacked2.sign = neg2tc(unpacked2.sign->type, unpacked2.sign);

  // figure out which operand has the bigger exponent
  const expr2tc exponent_difference = subtract_exponents(unpacked1, unpacked2);
  expr2tc src2_sign = sign_bit(exponent_difference);
  expr2tc src2_bigger = notequal2tc(gen_zero(src2_sign->type), src2_sign);

  const expr2tc bigger_exponent = if2tc(
    unpacked2.exponent->type,
    src2_bigger,
    unpacked2.exponent,
    unpacked1.exponent);

  // swap fractions as needed
  const expr2tc new_fraction1 = if2tc(
    unpacked2.fraction->type,
    src2_bigger,
    unpacked2.fraction,
    unpacked1.fraction);

  const expr2tc new_fraction2 = if2tc(
    unpacked1.fraction->type,
    src2_bigger,
    unpacked1.fraction,
    unpacked2.fraction);

  // compute distance
  const expr2tc distance = typecast2tc(
    type_pool.get_uint(spec.e),
    abs2tc(exponent_difference->type, exponent_difference));

  // limit the distance: shifting more than f+3 bits is unnecessary
  const expr2tc limited_dist = limit_distance(distance, spec.f + 3);

  // pad fractions with 3 zeros from below
  expr2tc three_zeros = gen_zero(type_pool.get_uint(3));

  // add 4 to spec.f because unpacked new_fraction has the hidden bit
  const expr2tc fraction1_padded =
    concat2tc(type_pool.get_uint(spec.f + 4), new_fraction1, three_zeros);

  const expr2tc fraction2_padded =
    concat2tc(type_pool.get_uint(spec.f + 4), new_fraction2, three_zeros);

  // shift new_fraction2
  expr2tc sticky_bit;
  const expr2tc fraction1_shifted = fraction1_padded;
  const expr2tc fraction2_shifted =
    sticky_right_shift(fraction2_padded, limited_dist, sticky_bit);

  // sticky bit: 'or' of the bits lost by the right-shift
  expr2tc fraction2_stickied = bitor2tc(
    fraction2_shifted->type,
    fraction2_shifted,
    concat2tc(
      fraction2_shifted->type,
      gen_zero(type_pool.get_uint(spec.f + 3)),
      sticky_bit));

  // need to have two extra fraction bits for addition and rounding
  const expr2tc fraction1_ext =
    typecast2tc(type_pool.get_uint(spec.f + 4 + 2), fraction1_shifted);
  const expr2tc fraction2_ext =
    typecast2tc(type_pool.get_uint(spec.f + 4 + 2), fraction2_stickied);

  unbiased_floatt result;

  // now add/sub them
  expr2tc subtract_lit = notequal2tc(unpacked1.sign, unpacked2.sign);

  result.fraction = if2tc(
    fraction1_ext->type,
    subtract_lit,
    sub2tc(fraction1_ext->type, fraction1_ext, fraction2_ext),
    add2tc(fraction1_ext->type, fraction1_ext, fraction2_ext));

  // sign of result
  std::size_t width = result.fraction->type->get_width();
  expr2tc fraction_sign = sign_bit(result.fraction);
  result.fraction = typecast2tc(
    type_pool.get_int(width),
    abs2tc(
      result.fraction->type,
      typecast2tc(type_pool.get_int(width), result.fraction)));

  result.exponent = bigger_exponent;

  // adjust the exponent for the fact that we added two bits to the fraction
  result.exponent = add2tc(
    type_pool.get_int(spec.e + 1),
    typecast2tc(type_pool.get_int(spec.e + 1), result.exponent),
    from_integer(2, type_pool.get_int(spec.e + 1)));

  // NaN?
  result.NaN = or2tc(
    and2tc(
      and2tc(unpacked1.infinity, unpacked2.infinity),
      notequal2tc(unpacked1.sign, unpacked2.sign)),
    or2tc(unpacked1.NaN, unpacked2.NaN));

  // infinity?
  result.infinity =
    and2tc(not2tc(result.NaN), or2tc(unpacked1.infinity, unpacked2.infinity));

  // zero?
  // Note that:
  //  1. The zero flag isn't used apart from in divide and
  //     is only set on unpack
  //  2. Subnormals mean that addition or subtraction can't round to 0,
  //     thus we can perform this test now
  //  3. The rules for sign are different for zero
  result.zero = and2tc(
    not2tc(or2tc(result.infinity, result.NaN)),
    equality2tc(result.fraction, gen_zero(result.fraction->type)));

  // sign
  expr2tc add_sub_sign = notequal2tc(
    if2tc(unpacked2.sign->type, src2_bigger, unpacked2.sign, unpacked1.sign),
    fraction_sign);

  expr2tc infinity_sign = if2tc(
    unpacked1.sign->type, unpacked1.infinity, unpacked1.sign, unpacked2.sign);

  rounding_mode_bitst rounding_mode_bits(rm);

  expr2tc zero_sign = if2tc(
    type_pool.get_bool(),
    rounding_mode_bits.round_to_minus_inf,
    or2tc(unpacked1.sign, unpacked2.sign),
    and2tc(unpacked1.sign, unpacked2.sign));

  result.sign = if2tc(
    infinity_sign->type,
    result.infinity,
    infinity_sign,
    if2tc(zero_sign->type, result.zero, zero_sign, add_sub_sign));

  return rounder(result, rm, spec);
}

/// Limits the shift distance
expr2tc float_bvt::limit_distance(const expr2tc &dist, mp_integer limit)
{
  std::size_t nb_bits = integer2unsigned(address_bits(limit));
  std::size_t dist_width = dist->type->get_width();

  if(dist_width <= nb_bits)
    return dist;

  expr2tc upper_bits = extract2tc(
    type_pool.get_uint(dist_width - nb_bits), dist, dist_width - 1, nb_bits);
  expr2tc upper_bits_zero = equality2tc(upper_bits, gen_zero(upper_bits->type));

  expr2tc lower_bits =
    extract2tc(type_pool.get_uint(nb_bits), dist, nb_bits - 1, 0);

  return if2tc(
    lower_bits->type,
    upper_bits_zero,
    lower_bits,
    from_integer(BigInt(ULONG_LONG_MAX), type_pool.get_uint(nb_bits)));
}

expr2tc float_bvt::mul(
  const expr2tc &src1,
  const expr2tc &src2,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  // unpack
  const unbiased_floatt unpacked1 = unpack(src1, spec);
  const unbiased_floatt unpacked2 = unpack(src2, spec);

  // zero-extend the fractions (unpacked fraction has the hidden bit)
  type2tc new_fraction_type = type_pool.get_uint((spec.f + 1) * 2);
  const expr2tc fraction1 = typecast2tc(new_fraction_type, unpacked1.fraction);
  const expr2tc fraction2 = typecast2tc(new_fraction_type, unpacked2.fraction);

  // multiply the fractions
  unbiased_floatt result;
  result.fraction = mul2tc(fraction1->type, fraction1, fraction2);

  // extend exponents to account for overflow
  // add two bits, as we do extra arithmetic on it later
  type2tc new_exponent_type = type_pool.get_int(spec.e + 2);
  const expr2tc exponent1 = typecast2tc(new_exponent_type, unpacked1.exponent);
  const expr2tc exponent2 = typecast2tc(new_exponent_type, unpacked2.exponent);

  expr2tc added_exponent = add2tc(exponent1->type, exponent1, exponent2);

  // Adjust exponent; we are thowing in an extra fraction bit,
  // it has been extended above.
  result.exponent =
    add2tc(added_exponent->type, added_exponent, gen_one(new_exponent_type));

  // new sign
  result.sign = notequal2tc(unpacked1.sign, unpacked2.sign);

  // infinity?
  result.infinity = or2tc(unpacked1.infinity, unpacked2.infinity);

  // NaN?
  {
    or2tc NaN_cond(isnan(src1, spec), isnan(src2, spec));

    // infinity * 0 is NaN!
    or2tc mult_zero_nan(
      and2tc(unpacked1.zero, unpacked2.infinity),
      and2tc(unpacked2.zero, unpacked1.infinity));

    result.NaN = or2tc(NaN_cond, mult_zero_nan);
  }

  return rounder(result, rm, spec);
}

expr2tc float_bvt::div(
  const expr2tc &src1,
  const expr2tc &src2,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  // unpack
  const unbiased_floatt unpacked1 = unpack(src1, spec);
  const unbiased_floatt unpacked2 = unpack(src2, spec);

  std::size_t fraction_width = unpacked1.fraction->type->get_width();
  std::size_t div_width = fraction_width * 2 + 1;

  // pad fraction1 with zeros
  expr2tc fraction1 = concat2tc(
    type_pool.get_uint(div_width),
    unpacked1.fraction,
    from_integer(0, type_pool.get_uint(div_width - fraction_width)));

  // zero-extend fraction2 to match faction1
  const expr2tc fraction2 = typecast2tc(fraction1->type, unpacked2.fraction);

  // divide fractions
  unbiased_floatt result;
  expr2tc rem;

  // the below should be merged somehow
  result.fraction = div2tc(fraction1->type, fraction1, fraction2);
  rem = modulus2tc(fraction1->type, fraction1, fraction2);

  // is there a remainder?
  expr2tc have_remainder = notequal2tc(rem, gen_zero(rem->type));

  // we throw this into the result, as least-significant bit,
  // to get the right rounding decision
  result.fraction = concat2tc(
    type_pool.get_uint(div_width + 1), result.fraction, have_remainder);

  // We will subtract the exponents;
  // to account for overflow, we add a bit.
  const expr2tc exponent1 =
    typecast2tc(type_pool.get_int(spec.e + 1), unpacked1.exponent);
  const expr2tc exponent2 =
    typecast2tc(type_pool.get_int(spec.e + 1), unpacked2.exponent);

  // subtract exponents
  expr2tc added_exponent = sub2tc(exponent1->type, exponent1, exponent2);

  // adjust, as we have thown in extra fraction bits
  result.exponent = add2tc(
    added_exponent->type,
    added_exponent,
    from_integer(spec.f, added_exponent->type));

  // new sign
  result.sign = notequal2tc(unpacked1.sign, unpacked2.sign);

  // Infinity? This happens when
  // 1) dividing a non-nan/non-zero by zero, or
  // 2) first operand is inf and second is non-nan and non-zero
  // In particular, inf/0=inf.
  result.infinity = or2tc(
    and2tc(
      not2tc(unpacked1.zero), and2tc(not2tc(unpacked1.NaN), unpacked2.zero)),
    and2tc(
      unpacked1.infinity,
      and2tc(not2tc(unpacked2.NaN), not2tc(unpacked2.zero))));

  // NaN?
  result.NaN = or2tc(
    unpacked1.NaN,
    or2tc(
      unpacked2.NaN,
      or2tc(
        and2tc(unpacked1.zero, unpacked2.zero),
        and2tc(unpacked1.infinity, unpacked2.infinity))));

  // Division by infinity produces zero, unless we have NaN
  expr2tc force_zero = and2tc(not2tc(unpacked1.NaN), unpacked2.infinity);

  result.fraction = if2tc(
    result.fraction->type,
    force_zero,
    gen_zero(result.fraction->type),
    result.fraction);

  return rounder(result, rm, spec);
}

expr2tc float_bvt::isinf(const expr2tc &src, const ieee_float_spect &spec)
{
  return and2tc(exponent_all_ones(src, spec), fraction_all_zeros(src, spec));
}

/// Gets the unbiased exponent in a floating-point bit-vector
expr2tc
float_bvt::get_exponent(const expr2tc &src, const ieee_float_spect &spec)
{
  return extract2tc(
    type_pool.get_uint(spec.e), src, spec.f + spec.e - 1, spec.f);
}

/// Gets the fraction without hidden bit in a floating-point bit-vector src
expr2tc
float_bvt::get_fraction(const expr2tc &src, const ieee_float_spect &spec)
{
  return extract2tc(type_pool.get_uint(spec.f), src, spec.f - 1, 0);
}

expr2tc float_bvt::isnan(const expr2tc &src, const ieee_float_spect &spec)
{
  return and2tc(
    exponent_all_ones(src, spec), not2tc(fraction_all_zeros(src, spec)));
}

/// normalize fraction/exponent pair returns 'zero' if fraction is zero
void float_bvt::normalization_shift(expr2tc &fraction, expr2tc &exponent)
{
  // n-log-n alignment shifter.
  // The worst-case shift is the number of fraction
  // bits minus one, in case the faction is one exactly.
  std::size_t fraction_bits = fraction->type->get_width();
  std::size_t exponent_bits = exponent->type->get_width();
  assert(fraction_bits != 0);

  unsigned depth = integer2unsigned(address_bits(fraction_bits - 1));

  if(exponent_bits < depth)
    exponent = typecast2tc(type_pool.get_int(depth), exponent);

  expr2tc exponent_delta = gen_zero(exponent->type);

  for(int d = depth - 1; d >= 0; d--)
  {
    unsigned distance = (1 << d);
    assert(fraction_bits > distance);

    // check if first 'distance'-many bits are zeros
    const extract2tc prefix(
      type_pool.get_uint(distance),
      fraction,
      fraction_bits - 1,
      fraction_bits - distance);

    const equality2tc prefix_is_zero(prefix, gen_zero(prefix->type));

    // If so, shift the zeros out left by 'distance'.
    // Otherwise, leave as is.

    const shl2tc shifted(
      fraction->type, fraction, from_integer(distance, get_int32_type()));

    fraction = if2tc(fraction->type, prefix_is_zero, shifted, fraction);

    // add corresponding weight to exponent
    assert(d < (signed int)exponent_bits);

    exponent_delta = bitor2tc(
      exponent_delta->type,
      exponent_delta,
      shl2tc(
        exponent_delta->type,
        typecast2tc(exponent_delta->type, prefix_is_zero),
        from_integer(distance, get_int32_type())));
  }

  exponent = sub2tc(exponent->type, exponent, exponent_delta);
}

/// make sure exponent is not too small; the exponent is unbiased
void float_bvt::denormalization_shift(
  expr2tc &fraction,
  expr2tc &exponent,
  const ieee_float_spect &spec)
{
  mp_integer bias = spec.bias();

  // Is the exponent strictly less than -bias+1, i.e., exponent<-bias+1?
  // This is transformed to distance=(-bias+1)-exponent
  // i.e., distance>0
  // Note that 1-bias is the exponent represented by 0...01,
  // i.e. the exponent of the smallest normal number and thus the 'base'
  // exponent for subnormal numbers.

  std::size_t exponent_bits = exponent->type->get_width();
  assert(exponent_bits >= spec.e);

  // Need to sign extend to avoid overflow.  Note that this is a
  // relatively rare problem as the value needs to be close to the top
  // of the exponent range and then range must not have been
  // previously extended as add, multiply, etc. do.  This is primarily
  // to handle casting down from larger ranges.
  exponent = typecast2tc(type_pool.get_int(exponent_bits + 1), exponent);

  expr2tc distance =
    sub2tc(exponent->type, from_integer(-bias + 1, exponent->type), exponent);

  // use sign bit
  expr2tc denormal = and2tc(
    not2tc(notequal2tc(gen_zero(type_pool.get_uint(1)), sign_bit(distance))),
    notequal2tc(distance, gen_zero(distance->type)));

  // Care must be taken to not loose information required for the
  // guard and sticky bits.  +3 is for the hidden, guard and sticky bits.
  std::size_t fraction_bits = fraction->type->get_width();

  if(fraction_bits < spec.f + 3)
  {
    // Add zeros at the LSB end for the guard bit to shift into
    fraction = concat2tc(
      type_pool.get_uint(spec.f + 3),
      fraction,
      gen_zero(type_pool.get_uint(spec.f + 3 - fraction_bits)));
  }

  expr2tc denormalisedFraction = fraction;

  expr2tc sticky_bit = gen_false_expr();
  denormalisedFraction = sticky_right_shift(fraction, distance, sticky_bit);

  denormalisedFraction = bitor2tc(
    denormalisedFraction->type,
    denormalisedFraction,
    typecast2tc(denormalisedFraction->type, sticky_bit));

  fraction =
    if2tc(denormalisedFraction->type, denormal, denormalisedFraction, fraction);

  exponent = if2tc(
    exponent->type, denormal, from_integer(-bias, exponent->type), exponent);
}

expr2tc float_bvt::rounder(
  const unbiased_floatt &src,
  const expr2tc &rm,
  const ieee_float_spect &spec)
{
  // incoming: some fraction (with explicit 1),
  //           some exponent without bias
  // outgoing: rounded, with right size, with hidden bit, bias

  expr2tc aligned_fraction = src.fraction, aligned_exponent = src.exponent;

  {
    std::size_t exponent_bits =
      std::max(
        (std::size_t)integer2size_t(address_bits(spec.f)),
        (std::size_t)spec.e) +
      1;

    // before normalization, make sure exponent is large enough
    if(aligned_exponent->type->get_width() < exponent_bits)
    {
      // sign extend
      aligned_exponent =
        typecast2tc(type_pool.get_int(exponent_bits), aligned_exponent);
    }
  }

  // align it!
  normalization_shift(aligned_fraction, aligned_exponent);
  denormalization_shift(aligned_fraction, aligned_exponent, spec);

  unbiased_floatt result;
  result.fraction = aligned_fraction;
  result.exponent = aligned_exponent;
  result.sign = src.sign;
  result.NaN = src.NaN;
  result.infinity = src.infinity;

  rounding_mode_bitst rounding_mode_bits(rm);
  round_fraction(result, rounding_mode_bits, spec);
  round_exponent(result, rounding_mode_bits, spec);

  return pack(bias(result, spec), spec);
}

/// rounding decision for fraction using sticky bit
expr2tc float_bvt::fraction_rounding_decision(
  const std::size_t dest_bits,
  const expr2tc sign,
  const expr2tc &fraction,
  const rounding_mode_bitst &rounding_mode_bits)
{
  std::size_t fraction_bits = fraction->type->get_width();

  assert(dest_bits < fraction_bits);

  // we have too many fraction bits
  std::size_t extra_bits = fraction_bits - dest_bits;

  // more than two extra bits are superflus, and are
  // turned into a sticky bit
  expr2tc sticky_bit = gen_false_expr();
  if(extra_bits >= 2)
  {
    // We keep most-significant bits, and thus the tail is made
    // of least-significant bits.
    expr2tc tail = extract2tc(
      type_pool.get_uint(extra_bits - 2 + 1), fraction, extra_bits - 2, 0);
    sticky_bit = notequal2tc(tail, gen_zero(tail->type));
  }

  // the rounding bit is the last extra bit
  assert(extra_bits >= 1);
  expr2tc rounding_bit =
    extract2tc(type_pool.get_uint(1), fraction, extra_bits - 1, extra_bits - 1);

  // we get one bit of the fraction for some rounding decisions
  expr2tc rounding_least =
    extract2tc(type_pool.get_uint(1), fraction, extra_bits, extra_bits);

  // round-to-nearest (ties to even)
  expr2tc round_to_even =
    and2tc(rounding_bit, or2tc(rounding_least, sticky_bit));

  // round up
  expr2tc round_to_plus_inf =
    and2tc(not2tc(sign), or2tc(rounding_bit, sticky_bit));

  // round down
  expr2tc round_to_minus_inf = and2tc(sign, or2tc(rounding_bit, sticky_bit));

  // round to zero
  expr2tc round_to_zero = gen_false_expr();

  // now select appropriate one
  return if2tc(
    round_to_even->type,
    rounding_mode_bits.round_to_even,
    round_to_even,
    if2tc(
      round_to_plus_inf->type,
      rounding_mode_bits.round_to_plus_inf,
      round_to_plus_inf,
      if2tc(
        round_to_minus_inf->type,
        rounding_mode_bits.round_to_minus_inf,
        round_to_minus_inf,
        if2tc(
          round_to_zero->type,
          rounding_mode_bits.round_to_zero,
          round_to_zero,
          gen_zero(round_to_zero->type))))); // otherwise zero
}

void float_bvt::round_fraction(
  unbiased_floatt &result,
  const rounding_mode_bitst &rounding_mode_bits,
  const ieee_float_spect &spec)
{
  std::size_t fraction_size = spec.f + 1;
  std::size_t result_fraction_size = result.fraction->type->get_width();

  // do we need to enlarge the fraction?
  if(result_fraction_size < fraction_size)
  {
    // pad with zeros at bottom
    std::size_t padding = fraction_size - result_fraction_size;

    result.fraction = concat2tc(
      type_pool.get_uint(fraction_size),
      result.fraction,
      gen_zero(type_pool.get_uint(padding)));
  }
  else if(result_fraction_size == fraction_size) // it stays
  {
    // do nothing
  }
  else // fraction gets smaller -- rounding
  {
    std::size_t extra_bits = result_fraction_size - fraction_size;
    assert(extra_bits >= 1);

    // this computes the rounding decision
    expr2tc increment = fraction_rounding_decision(
      fraction_size, result.sign, result.fraction, rounding_mode_bits);

    // chop off all the extra bits
    result.fraction = extract2tc(
      type_pool.get_uint(fraction_size),
      result.fraction,
      result_fraction_size - 1,
      extra_bits);

    // When incrementing due to rounding there are two edge
    // cases we need to be aware of:
    //  1. If the number is normal, the increment can overflow.
    //     In this case we need to increment the exponent and
    //     set the MSB of the fraction to 1.
    //  2. If the number is the largest subnormal, the increment
    //     can change the MSB making it normal.  Thus the exponent
    //     must be incremented but the fraction will be OK.
    expr2tc oldMSB = extract2tc(
      type_pool.get_uint(1),
      result.fraction,
      fraction_size - 1,
      fraction_size - 1);

    // increment if 'increment' is true
    result.fraction = add2tc(
      result.fraction->type,
      result.fraction,
      typecast2tc(result.fraction->type, increment));

    // Normal overflow when old MSB == 1 and new MSB == 0
    expr2tc newMSB = extract2tc(
      type_pool.get_uint(1),
      result.fraction,
      fraction_size - 1,
      fraction_size - 1);

    expr2tc overflow = and2tc(oldMSB, not2tc(newMSB));

    // Subnormal to normal transition when old MSB == 0 and new MSB == 1
    expr2tc subnormal_to_normal = and2tc(not2tc(oldMSB), newMSB);

    // In case of an overflow or subnormal to normal conversion,
    // the exponent has to be incremented.
    result.exponent = add2tc(
      result.exponent->type,
      result.exponent,
      if2tc(
        result.exponent->type,
        or2tc(overflow, subnormal_to_normal),
        gen_one(result.exponent->type),
        gen_zero(result.exponent->type)));

    // post normalization of the fraction
    // In the case of overflow, set the MSB to 1
    // The subnormal case will have (only) the MSB set to 1
    result.fraction = bitor2tc(
      result.fraction->type,
      result.fraction,
      if2tc(
        result.fraction->type,
        overflow,
        from_integer(1 << (fraction_size - 1), result.fraction->type),
        gen_zero(result.fraction->type)));
  }
}

void float_bvt::round_exponent(
  unbiased_floatt &result,
  const rounding_mode_bitst &rounding_mode_bits,
  const ieee_float_spect &spec)
{
  std::size_t result_exponent_size = result.exponent->type->get_width();

  // do we need to enlarge the exponent?
  if(result_exponent_size < spec.e)
  {
    // should have been done before
    assert(false);
  }
  else if(result_exponent_size == spec.e) // it stays
  {
    // do nothing
  }
  else // exponent gets smaller -- chop off top bits
  {
    expr2tc old_exponent = result.exponent;
    result.exponent =
      extract2tc(type_pool.get_int(spec.e), result.exponent, spec.e - 1, 0);

    // max_exponent is the maximum representable
    // i.e. 1 higher than the maximum possible for a normal number
    expr2tc max_exponent =
      from_integer(spec.max_exponent() - spec.bias(), old_exponent->type);

    // the exponent is garbage if the fractional is zero

    expr2tc exponent_too_large = and2tc(
      greaterthanequal2tc(old_exponent, max_exponent),
      notequal2tc(result.fraction, gen_zero(result.fraction->type)));

    // Directed rounding modes round overflow to the maximum normal
    // depending on the particular mode and the sign
    expr2tc overflow_to_inf = or2tc(
      rounding_mode_bits.round_to_even,
      or2tc(
        and2tc(rounding_mode_bits.round_to_plus_inf, not2tc(result.sign)),
        and2tc(rounding_mode_bits.round_to_minus_inf, result.sign)));

    expr2tc set_to_max = and2tc(exponent_too_large, not2tc(overflow_to_inf));

    expr2tc largest_normal_exponent = from_integer(
      spec.max_exponent() - (spec.bias() + 1), result.exponent->type);

    result.exponent = if2tc(
      result.exponent->type,
      set_to_max,
      largest_normal_exponent,
      result.exponent);

    result.fraction = if2tc(
      result.fraction->type,
      set_to_max,
      from_integer(BigInt(ULONG_LONG_MAX), result.fraction->type),
      result.fraction);

    result.infinity =
      or2tc(result.infinity, and2tc(exponent_too_large, overflow_to_inf));
  }
}

/// takes an unbiased float, and applies the bias
float_bvt::biased_floatt
float_bvt::bias(const unbiased_floatt &src, const ieee_float_spect &spec)
{
  biased_floatt result;

  result.sign = src.sign;
  result.NaN = src.NaN;
  result.infinity = src.infinity;

  // we need to bias the new exponent
  result.exponent = add_bias(src.exponent, spec);

  // strip off the hidden bit
  assert(src.fraction->type->get_width() == spec.f + 1);

  expr2tc hidden_bit =
    extract2tc(type_pool.get_uint(1), src.fraction, spec.f, spec.f);
  expr2tc denormal = not2tc(hidden_bit);

  result.fraction =
    extract2tc(type_pool.get_uint(spec.f), src.fraction, spec.f - 1, 0);

  // make exponent zero if its denormal
  // (includes zero)
  result.exponent = if2tc(
    result.exponent->type,
    denormal,
    gen_zero(result.exponent->type),
    result.exponent);

  return result;
}

expr2tc float_bvt::add_bias(const expr2tc &src, const ieee_float_spect &spec)
{
  type2tc t = type_pool.get_uint(spec.e);
  return add2tc(t, typecast2tc(t, src), from_integer(spec.bias(), t));
}

expr2tc float_bvt::sub_bias(const expr2tc &src, const ieee_float_spect &spec)
{
  type2tc t = type_pool.get_int(spec.e);
  return sub2tc(t, typecast2tc(t, src), from_integer(spec.bias(), t));
}

float_bvt::unbiased_floatt
float_bvt::unpack(const expr2tc &src, const ieee_float_spect &spec)
{
  unbiased_floatt result;

  result.sign = sign_bit(src);

  result.fraction = get_fraction(src, spec);

  // add hidden bit
  expr2tc hidden_bit = isnormal(src, spec);
  result.fraction =
    concat2tc(type_pool.get_uint(spec.f + 1), hidden_bit, result.fraction);

  result.exponent = get_exponent(src, spec);

  // unbias the exponent
  expr2tc denormal = exponent_all_zeros(src, spec);

  result.exponent = if2tc(
    type_pool.get_int(spec.e),
    denormal,
    from_integer(-spec.bias() + 1, type_pool.get_int(spec.e)),
    sub_bias(result.exponent, spec));

  result.infinity = isinf(src, spec);
  result.zero = is_zero(src, spec);
  result.NaN = isnan(src, spec);

  return result;
}

expr2tc float_bvt::pack(const biased_floatt &src, const ieee_float_spect &spec)
{
  assert(src.fraction->type->get_width() == spec.f);
  assert(src.exponent->type->get_width() == spec.e);

  // do sign -- we make this 'false' for NaN
  expr2tc sign_bit =
    if2tc(src.sign->type, src.NaN, gen_zero(src.sign->type), src.sign);

  // the fraction is zero in case of infinity,
  // and one in case of NaN
  expr2tc fraction = if2tc(
    src.fraction->type,
    src.NaN,
    gen_one(src.fraction->type),
    if2tc(
      src.fraction->type,
      src.infinity,
      gen_zero(src.fraction->type),
      src.fraction));

  expr2tc infinity_or_NaN = or2tc(src.NaN, src.infinity);

  // do exponent
  expr2tc exponent = if2tc(
    src.exponent->type,
    infinity_or_NaN,
    from_integer(-1, src.exponent->type),
    src.exponent);

  // stitch all three together
  return concat2tc(
    floatbv_type2tc(spec.f, spec.e),
    sign_bit,
    concat2tc(type_pool.get_uint(spec.e + spec.f), exponent, fraction));
}

expr2tc float_bvt::sticky_right_shift(
  const expr2tc &op,
  const expr2tc &dist,
  expr2tc &sticky)
{
  std::size_t d = 1, width = op->type->get_width();
  expr2tc result = op;
  sticky = gen_false_expr();

  std::size_t dist_width = dist->type->get_width();

  for(std::size_t stage = 0; stage < dist_width; stage++)
  {
    expr2tc tmp =
      lshr2tc(result->type, result, from_integer(d, get_int32_type()));

    expr2tc lost_bits;
    if(d <= width)
      lost_bits = extract2tc(type_pool.get_uint(d), result, d - 1, 0);
    else
      lost_bits = result;

    expr2tc dist_bit = extract2tc(dist->type, dist, stage, stage);

    sticky = or2tc(
      and2tc(dist_bit, notequal2tc(lost_bits, gen_zero(lost_bits->type))),
      sticky);

    result = if2tc(tmp->type, dist_bit, tmp, result);

    d = d << 1;
  }

  return result;
}
