/*
 * fp_conv.h
 *
 *  Created on: Mar 3, 2017
 *      Author: mramalho
 */

#ifndef SOLVERS_SMT_FP_CONV_H_
#define SOLVERS_SMT_FP_CONV_H_

#include "smt_conv.h"

class fp_convt
{
public:
  fp_convt(smt_convt *_ctx);
  virtual ~fp_convt() = default;

  /** Create a floating point bitvector
   *  @param thereal the ieee float number
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw);

  /** Create a NaN floating point bitvector
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);

  /** Create a (+/-)inf floating point bitvector
   *  @param sgn Whether this bitvector is negative or positive.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);

  /** Create a rounding mode to be used by floating point cast and arith ops
   *  @param rm the kind of rounding mode
   *  @return The newly created rounding mode smt_ast. */
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);

  /** Typecast from a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);

  /** Typecast to a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);

  /** Calculate the nearby int from a floating point, considering the rounding mode
   *  @param expr the nearby int expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);

  /** Convert the ieee arithmetic operations (add, sub, mul, div, mod)
   *  @param expr the arithmetic operations
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);

  /** Convert the ieee arithmetic fused-multiply add (fma)
   *  @param expr the fma operation
   *  @return The newly created fma smt_ast */
  virtual smt_astt mk_smt_bvfloat_fma(const expr2tc &expr);

  /** Create a sort representing a floating-point number.
   *  @param type The floating-point type.
   *  @return The floating-point representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_bvfloat_sort(const unsigned ew, const unsigned sw);

  smt_convt *ctx;
};

#endif /* SOLVERS_SMT_FP_CONV_H_ */
