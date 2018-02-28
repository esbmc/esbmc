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
   *  @param thereal the floating-point number
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_fpbv(const ieee_floatt &thereal);

  /** Create a sort representing a floating-point number.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The floating-point representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);

  /** Create a sort representing a floating-point rounding mode.
   *  @return The floating-point rounding mode representation of the type,
   *  wrapped in an smt_sort. */
  virtual smt_sortt mk_fpbv_rm_sort();

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
  virtual smt_astt
  mk_smt_typecast_from_fpbv_to_ubv(expr2tc from, std::size_t width);

  /** Typecast from a floating point
   *  @param from the floating point being casted to signed bitvector
   *  @param to the signed bitvector resulting type
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_from_fpbv_to_sbv(expr2tc from, std::size_t width);

  /** Typecast from a floating point
   *  @param from the floating point being casted to floating-point
   *  @param to the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_from_fpbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm);

  /** Typecast to a floating point
   *  @param from the unsigned bitvector being casted to a floating-point
   *  @param cast the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_ubv_to_fpbv(expr2tc from, type2tc to, expr2tc rm);

  /** Typecast to a floating point
   *  @param from the signed bitvector being casted to a floating-point
   *  @param cast the floating-point resulting type
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt
  mk_smt_typecast_sbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm);

  /** Calculate the nearby int from a floating point, considering the rounding mode
   *  @param from the floating-point
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_nearbyint_from_float(expr2tc from, expr2tc rm);

  /** Convert a ieee addition
   *  @param lhs left hand side of the addition
   *  @param rhs right hand side of the addition
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_add(expr2tc lhs, expr2tc rhs, expr2tc rm);

  /** Convert a ieee subtraction
   *  @param lhs left hand side of the subtraction
   *  @param rhs right hand side of the subtraction
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_sub(expr2tc lhs, expr2tc rhs, expr2tc rm);

  /** Convert a ieee multiplication
   *  @param lhs left hand side of the multiplication
   *  @param rhs right hand side of the multiplication
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_mul(expr2tc lhs, expr2tc rhs, expr2tc rm);

  /** Convert the ieee division
   *  @param lhs left hand side of the division
   *  @param rhs right hand side of the division
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_fpbv_div(expr2tc lhs, expr2tc rhs, expr2tc rm);

  /** Convert the ieee arithmetic square-root (sqrt)
   *  @param op the sqrt radicand
   *  @param rm the rounding mode
   *  @return The newly created sqrt smt_ast */
  virtual smt_astt mk_smt_fpbv_sqrt(expr2tc rd, expr2tc rm);

  /** Convert the ieee arithmetic fused-multiply add (fma): round((v1 * v2) + v3)
   *  @param v1 in the equation
   *  @param v2 in the equation
   *  @param v3 in the equation
   *  @param rm the rounding mode
   *  @return The newly created fma smt_ast */
  virtual smt_astt
  mk_smt_fpbv_fma(expr2tc v1, expr2tc v2, expr2tc v3, expr2tc rm);

  /** Convert an ieee equality
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.eq smt_ast. */
  virtual smt_astt mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee greater than
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.gt smt_ast. */
  virtual smt_astt mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee less than
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.lt smt_ast. */
  virtual smt_astt mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee greater than or equal
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.gt smt_ast. */
  virtual smt_astt mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee less than or equal
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.lt smt_ast. */
  virtual smt_astt mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee is_nan operation
   *  @param op the operand
   *  @return The newly created fp.isNaN smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_nan(smt_astt op);

  /** Convert an ieee is_inf operation
   *  @param op the operand
   *  @return The newly created fp.isInfinite smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_inf(smt_astt op);

  /** Convert an ieee is_normal operation
   *  @param op the operand
   *  @return The newly created fp.isNormal smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_normal(smt_astt op);

  /** Convert an ieee is_zero operation
   *  @param op the operand
   *  @return The newly created fp.isZero smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_zero(smt_astt op);

  /** Convert an ieee is_neg operation
   *  @param op the operand
   *  @return The newly created fp.isNegative smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_negative(smt_astt op);

  /** Convert an ieee is_pos operation
   *  @param op the operand
   *  @return The newly created fp.isPositive smt_ast. */
  virtual smt_astt mk_smt_fpbv_is_positive(smt_astt op);

  /** Convert an ieee abs operation
   *  @param op the operand
   *  @return The newly created fp.abs smt_ast. */
  virtual smt_astt mk_smt_fpbv_abs(smt_astt op);

  /** Extract the assignment to a floating-point from the SMT solvers model.
   *  @param a the AST whos value we wish to know.
   *  @return the ieee floating-point */
  virtual ieee_floatt get_fpbv(smt_astt a);

  smt_convt *ctx;
};

#endif /* SOLVERS_SMT_FP_CONV_H_ */
