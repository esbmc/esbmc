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
  virtual smt_astt mk_smt_fpbv(const ieee_floatt &thereal);

  /** Create a NaN floating point bitvector
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw);

  /** Create a (+/-)inf floating point bitvector
   *  @param sgn Whether this bitvector is negative or positive.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw);

  /** Create a rounding mode to be used by floating point cast and arith ops
   *  @param rm the kind of rounding mode
   *  @return The newly created rounding mode smt_ast. */
  virtual smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm);

  /** Typecast from a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_from_fpbv(const typecast2t &cast);

  /** Typecast to a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_to_fpbv(const typecast2t &cast);

  /** Calculate the nearby int from a floating point, considering the rounding mode
   *  @param expr the nearby int expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);

  /** Convert the ieee arithmetic operations (add, sub, mul, div, mod)
   *  @param expr the arithmetic operations
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_arith_ops(const expr2tc &expr);

  /** Convert the ieee arithmetic fused-multiply add (fma)
   *  @param expr the fma operation
   *  @return The newly created fma smt_ast */
  virtual smt_astt mk_smt_fpbv_fma(const expr2tc &expr);

  /** Create a sort representing a floating-point number.
   *  @param type The floating-point type.
   *  @return The floating-point representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);

  /** Extract the assignment to a floating-point from the SMT solvers model.
   *  @param t The AST type
   *  @param a The AST whos value we wish to know.
   *  @return Expression representation of a's value, as a constant_floatbv2tc */
  virtual expr2tc get_fpbv(const type2tc &t, smt_astt a);

  smt_convt *ctx;
};

#endif /* SOLVERS_SMT_FP_CONV_H_ */
