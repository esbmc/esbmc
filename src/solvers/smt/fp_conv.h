/*
 * fp_conv.h
 *
 *  Created on: Mar 3, 2017
 *      Author: mramalho
 */

#ifndef SOLVERS_SMT_FP_CONV_H_
#define SOLVERS_SMT_FP_CONV_H_

#include <solvers/smt/smt_ast.h>
#include <solvers/smt/smt_sort.h>

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
   *  @param from the floating point being casted to unsigned bitvector
   *  @param to the unsigned bitvector resulting type
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_from_fpbv_to_ubv(
    smt_astt from,
    smt_sortt to);

  /** Typecast from a floating point
   *  @param from the floating point being casted to signed bitvector
   *  @param to the signed bitvector resulting type
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_from_fpbv_to_sbv(
    smt_astt from,
    smt_sortt to);

  /** Typecast from a floating point
   *  @param from the floating point being casted to floating-point
   *  @param to the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_from_fpbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);

  /** Typecast to a floating point
   *  @param from the unsigned bitvector being casted to a floating-point
   *  @param cast the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);

  /** Typecast to a floating point
   *  @param from the signed bitvector being casted to a floating-point
   *  @param cast the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);

  /** Calculate the nearby int from a floating point, considering the rounding mode
   *  @param from the floating-point
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm);

  /** Convert a ieee addition
   *  @param lhs left hand side of the addition
   *  @param rhs right hand side of the addition
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert a ieee subtraction
   *  @param lhs left hand side of the subtraction
   *  @param rhs right hand side of the subtraction
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert a ieee multiplication
   *  @param lhs left hand side of the multiplication
   *  @param rhs right hand side of the multiplication
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert the ieee division
   *  @param lhs left hand side of the division
   *  @param rhs right hand side of the division
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert the ieee arithmetic square-root (sqrt)
   *  @param op the sqrt radicand
   *  @param rm the rounding mode
   *  @return The newly created sqrt smt_ast */
  virtual smt_astt mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm);

  /** Convert the ieee arithmetic fused-multiply add (fma): round((v1 * v2) + v3)
   *  @param v1 in the equation
   *  @param v2 in the equation
   *  @param v3 in the equation
   *  @param rm the rounding mode
   *  @return The newly created fma smt_ast */
  virtual smt_astt
  mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm);

  /** Create a sort representing a floating-point number.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The floating-point representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);

  /** Create a sort representing a floating-point rounding mode.
   *  @return The floating-point rounding mode representation of the type,
   *  wrapped in an smt_sort. */
  virtual smt_sortt mk_fpbv_rm_sort();

  /** Extract the assignment to a floating-point from the SMT solvers model.
   *  @param t The AST type
   *  @param a The AST whos value we wish to know.
   *  @return Expression representation of a's value, as a constant_floatbv2tc */
  virtual expr2tc get_fpbv(const type2tc &t, smt_astt a);

  smt_convt *ctx;
};

#endif /* SOLVERS_SMT_FP_CONV_H_ */
