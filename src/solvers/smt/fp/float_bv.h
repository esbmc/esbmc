/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SOLVERS_FLOATBV_FLOAT_BV_H
#define CPROVER_SOLVERS_FLOATBV_FLOAT_BV_H

#include <util/irep2_utils.h>

class float_bvt
{
public:
  // add/sub
  static expr2tc add_sub(
    bool subtract,
    const expr2tc &op0,
    const expr2tc &op1,
    const expr2tc &rm,
    const ieee_float_spect &spec);

  // mul/div
  static expr2tc mul(
    const expr2tc &,
    const expr2tc &,
    const expr2tc &rm,
    const ieee_float_spect &);
  static expr2tc div(
    const expr2tc &,
    const expr2tc &,
    const expr2tc &rm,
    const ieee_float_spect &);

  // conversion
  static expr2tc from_unsigned_integer(
    const expr2tc &,
    const expr2tc &rm,
    const ieee_float_spect &);
  static expr2tc from_signed_integer(
    const expr2tc &,
    const expr2tc &rm,
    const ieee_float_spect &);
  static expr2tc to_signed_integer(
    const expr2tc &src,
    std::size_t dest_width,
    const ieee_float_spect &);
  static expr2tc to_unsigned_integer(
    const expr2tc &src,
    std::size_t dest_width,
    const ieee_float_spect &);
  static expr2tc to_integer(
    const expr2tc &src,
    std::size_t dest_width,
    bool is_signed,
    const ieee_float_spect &);
  static expr2tc conversion(
    const expr2tc &src,
    const expr2tc &rm,
    const ieee_float_spect &src_spec,
    const ieee_float_spect &dest_spec);

  // helpers
  static ieee_float_spect get_spec(const expr2tc &);

protected:
  static expr2tc is_zero(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc isnan(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc isinf(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc isnormal(const expr2tc &src, const ieee_float_spect &spec);

  // still biased
  static expr2tc get_exponent(const expr2tc &src, const ieee_float_spect &spec);

  // without hidden bit
  static expr2tc get_fraction(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc sign_bit(const expr2tc &op);

  static expr2tc
  exponent_all_ones(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc
  exponent_all_zeros(const expr2tc &src, const ieee_float_spect &spec);
  static expr2tc
  fraction_all_zeros(const expr2tc &src, const ieee_float_spect &spec);

  struct rounding_mode_bitst
  {
  public:
    // these are mutually exclusive, obviously
    expr2tc round_to_even;
    expr2tc round_to_zero;
    expr2tc round_to_plus_inf;
    expr2tc round_to_minus_inf;

    void get(const expr2tc &rm);
    explicit rounding_mode_bitst(const expr2tc &rm)
    {
      get(rm);
    }
  };

  // unpacked
  static void normalization_shift(expr2tc &fraction, expr2tc &exponent);
  static void denormalization_shift(
    expr2tc &fraction,
    expr2tc &exponent,
    const ieee_float_spect &);

  static expr2tc add_bias(const expr2tc &exponent, const ieee_float_spect &);
  static expr2tc sub_bias(const expr2tc &exponent, const ieee_float_spect &);

  static expr2tc limit_distance(const expr2tc &dist, mp_integer limit);

  struct unpacked_floatt
  {
    expr2tc sign, infinity, zero, NaN;
    expr2tc fraction, exponent;

    unpacked_floatt()
      : sign(gen_false_expr()),
        infinity(gen_false_expr()),
        zero(gen_false_expr()),
        NaN(gen_false_expr())
    {
    }

    void dump()
    {
      sign->dump();
      infinity->dump();
      zero->dump();
      NaN->dump();
      fraction->dump();
      exponent->dump();
    }
  };

  // This has a biased exponent (unsigned)
  // and an _implicit_ hidden bit.
  struct biased_floatt : public unpacked_floatt
  {
  };

  // The hidden bit is explicit,
  // and the exponent is not biased (signed)
  struct unbiased_floatt : public unpacked_floatt
  {
  };

  static biased_floatt bias(const unbiased_floatt &, const ieee_float_spect &);

  // this takes unpacked format, and returns packed
  static expr2tc rounder(
    const unbiased_floatt &src,
    const expr2tc &rm,
    const ieee_float_spect &spec);
  static expr2tc pack(const biased_floatt &, const ieee_float_spect &);
  static unbiased_floatt
  unpack(const expr2tc &src, const ieee_float_spect &spec);

  static void round_fraction(
    unbiased_floatt &result,
    const rounding_mode_bitst &,
    const ieee_float_spect &);
  static void round_exponent(
    unbiased_floatt &result,
    const rounding_mode_bitst &,
    const ieee_float_spect &);

  // rounding decision for fraction
  static expr2tc fraction_rounding_decision(
    const std::size_t dest_bits,
    const expr2tc sign,
    const expr2tc &fraction,
    const rounding_mode_bitst &);

  // helpers for adder

  // computes src1.exponent-src2.exponent with extension
  static expr2tc
  subtract_exponents(const unbiased_floatt &src1, const unbiased_floatt &src2);

  // computes the "sticky-bit"
  static expr2tc
  sticky_right_shift(const expr2tc &op, const expr2tc &dist, expr2tc &sticky);
};

#endif // CPROVER_SOLVERS_FLOATBV_FLOAT_BV_H
