#include <solvers/smt/smt_conv.h>

smt_astt smt_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);

  expr2tc zero = gen_zero(opers.side_1->type);

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed =
    (is_signedbv_type(opers.side_1) || is_signedbv_type(opers.side_2));

  switch (overflow.operand->expr_id)
  {
  case expr2t::add_id:
  {
    if (is_signed && !int_encoding)
    {
      // Two cases: pos/pos, and neg/neg, which can over and underflow resp.
      // In pos/neg cases, no overflow or underflow is possible, for any value.
      expr2tc op1pos = lessthan2tc(zero, opers.side_1);
      expr2tc op2pos = lessthan2tc(zero, opers.side_2);
      expr2tc both_pos = and2tc(op1pos, op2pos);

      expr2tc negop1 = not2tc(op1pos);
      expr2tc negop2 = not2tc(op2pos);
      expr2tc both_neg = and2tc(negop1, negop2);

      expr2tc nooverflow =
        implies2tc(both_pos, greaterthanequal2tc(overflow.operand, zero));
      expr2tc nounderflow =
        implies2tc(both_neg, lessthanequal2tc(overflow.operand, zero));
      return convert_ast(not2tc(and2tc(nooverflow, nounderflow)));
    }
    else if (int_encoding)
    {
      // Get the width of the integer type
      auto const width = opers.side_1->type->get_width();

      // Extract the operand of the overflow expression
      const overflow2t &overflow_expr = to_overflow2t(expr);
      expr2tc result_expr = overflow_expr.operand;

      if (is_signed)
      {
        BigInt max_val = BigInt::power2(width - 1) - 1; // MAX_INT
        BigInt min_val = -BigInt::power2(width - 1);    // MIN_INT

        expr2tc max_int = constant_int2tc(opers.side_1->type, max_val);
        expr2tc min_int = constant_int2tc(opers.side_1->type, min_val);

        // Two cases: positive overflow and negative underflow
        expr2tc op1pos = lessthan2tc(zero, opers.side_1);
        expr2tc op2pos = lessthan2tc(zero, opers.side_2);
        expr2tc both_pos = and2tc(op1pos, op2pos);

        expr2tc negop1 = lessthan2tc(opers.side_1, zero);
        expr2tc negop2 = lessthan2tc(opers.side_2, zero);
        expr2tc both_neg = and2tc(negop1, negop2);

        // Overflow: if both are positive and result > MAX_INT
        expr2tc overflow = greaterthan2tc(result_expr, max_int);
        expr2tc pos_overflow = and2tc(both_pos, overflow);

        // Underflow: if both are negative and result < MIN_INT
        expr2tc underflow = lessthan2tc(result_expr, min_int);
        expr2tc neg_underflow = and2tc(both_neg, underflow);

        // If either overflow or underflow occurs, return true
        expr2tc overflow_check = or2tc(pos_overflow, neg_underflow);
        return convert_ast(overflow_check);
      }
      else
      {
        // Unsigned integer overflow detection
        BigInt max_val = BigInt::power2(width) - 1; // UINT_MAX

        expr2tc max_uint = constant_int2tc(opers.side_1->type, max_val);

        // Overflow occurs if result > UINT_MAX
        expr2tc overflow = greaterthan2tc(result_expr, max_uint);

        return convert_ast(overflow);
      }
    }

    // Just ensure the result is >= both operands.
    expr2tc ge1 = greaterthanequal2tc(overflow.operand, opers.side_1);
    expr2tc ge2 = greaterthanequal2tc(overflow.operand, opers.side_2);
    expr2tc res = and2tc(ge1, ge2);
    expr2tc inv = not2tc(res);
    return convert_ast(inv);
  }

  case expr2t::sub_id:
  {
    if (is_signed)
    {
      // Define zero constant for comparisons
      expr2tc zero = constant_int2tc(opers.side_1->type, 0);

      // Compute the subtraction result
      expr2tc sub_result =
        sub2tc(opers.side_1->type, opers.side_1, opers.side_2);

      // Overflow condition: (a > 0 && b < 0 && result < 0) || (a < 0 && b > 0 && result > 0)
      expr2tc a_pos = greaterthan2tc(opers.side_1, zero); // a > 0
      expr2tc b_neg = lessthan2tc(opers.side_2, zero);    // b < 0
      expr2tc result_neg = lessthan2tc(sub_result, zero); // result < 0
      expr2tc pos_minus_neg_overflow = and2tc(a_pos, and2tc(b_neg, result_neg));

      expr2tc a_neg = lessthan2tc(opers.side_1, zero);       // a < 0
      expr2tc b_pos = greaterthan2tc(opers.side_2, zero);    // b > 0
      expr2tc result_pos = greaterthan2tc(sub_result, zero); // result > 0
      expr2tc neg_minus_pos_overflow = and2tc(a_neg, and2tc(b_pos, result_pos));

      // Additional overflow checks for integer encoding
      if (int_encoding)
      {
        // Get the width of the integer type
        auto const width = opers.side_1->type->get_width();

        // Define minimum and maximum values for signed integers
        BigInt max_val = BigInt::power2(width - 1) - 1; // MAX_INT
        BigInt min_val = -BigInt::power2(width - 1);    // MIN_INT

        expr2tc max_int = constant_int2tc(opers.side_1->type, max_val);
        expr2tc min_int = constant_int2tc(opers.side_1->type, min_val);

        // Overflow occurs if result > MAX_INT or result < MIN_INT
        expr2tc overflow = greaterthan2tc(sub_result, max_int);
        expr2tc underflow = lessthan2tc(sub_result, min_int);
        expr2tc overflow_check = or2tc(overflow, underflow);

        // Combine overflow conditions
        expr2tc full_overflow_check = or2tc(
          or2tc(pos_minus_neg_overflow, neg_minus_pos_overflow),
          overflow_check);

        return convert_ast(full_overflow_check);
      }

      // Combine overflow conditions
      expr2tc overflow_detected =
        or2tc(pos_minus_neg_overflow, neg_minus_pos_overflow);

      return convert_ast(overflow_detected);
    }

    // Just ensure the result is <= the first operand.
    expr2tc sub = sub2tc(opers.side_1->type, opers.side_1, opers.side_2);
    expr2tc le = lessthanequal2tc(sub, opers.side_1);
    expr2tc inv = not2tc(le);
    return convert_ast(inv);
  }

  case expr2t::div_id:
  case expr2t::modulus_id:
  {
    if (is_signed)
    {
      // Handle signed division/modulus overflow cases
      // Dividing the most negative integer (MIN_INT) by -1 causes overflow
      BigInt topbit = -BigInt::power2(opers.side_1->type->get_width() - 1);
      expr2tc min_int = constant_int2tc(opers.side_1->type, topbit);
      expr2tc is_min_int = equality2tc(min_int, opers.side_1);

      // If MIN_INT is divided by -1, overflow occurs
      expr2tc minus_one = constant_int2tc(opers.side_1->type, -BigInt(1));
      expr2tc is_minus_one = equality2tc(minus_one, opers.side_2);

      // Return overflow condition for signed division
      return convert_ast(and2tc(is_minus_one, is_min_int));
    }

    // Detect unsigned integer overflow for division and modulus
    // Overflow occurs when dividing by zero
    expr2tc is_div_by_zero = equality2tc(opers.side_2, zero);

    // Overflow occurs if the dividend is greater than the maximum representable value
    expr2tc max_unsigned = constant_int2tc(
      opers.side_1->type, BigInt::power2(opers.side_1->type->get_width()) - 1);
    expr2tc is_overflow = greaterthan2tc(opers.side_1, max_unsigned);

    // Return overflow condition for unsigned division/modulus
    return convert_ast(or2tc(is_div_by_zero, is_overflow));
  }

  case expr2t::shl_id:
  case expr2t::mul_id:
  {
    // Zero extend; multiply; Make a decision based on the top half.
    unsigned int sz = zero->type->get_width();

    smt_astt arg1_ext = convert_ast(opers.side_1);
    smt_astt arg2_ext;

    if (!int_encoding)
      arg1_ext = is_signedbv_type(opers.side_1) ? mk_sign_ext(arg1_ext, sz)
                                                : mk_zero_ext(arg1_ext, sz);

    expr2tc op2 = opers.side_2;
    if (is_shl2t(overflow.operand))
      if (opers.side_1->type->get_width() != opers.side_2->type->get_width())
        op2 = typecast2tc(opers.side_1->type, opers.side_2);

    arg2_ext = convert_ast(op2);

    if (!int_encoding)
      arg2_ext = is_signedbv_type(op2) ? mk_sign_ext(arg2_ext, sz)
                                       : mk_zero_ext(arg2_ext, sz);
    smt_astt result;
    if (int_encoding)
    {
      // If using int_encoding, use mk_mul and mk_shl for multiplication and shift left
      result = is_mul2t(overflow.operand)
                 ? mk_mul(arg1_ext, arg2_ext)  // Use mk_mul for multiplication
                 : mk_shl(arg1_ext, arg2_ext); // Use mk_shl for shift left
    }
    else
    {
      // If not using int_encoding, fallback to original behavior (bvmul and bvshl)
      result = is_mul2t(overflow.operand) ? mk_bvmul(arg1_ext, arg2_ext)
                                          : mk_bvshl(arg1_ext, arg2_ext);
    }

    if (is_signed && !int_encoding)
    {
      // Extract top half plus one (for the sign)
      smt_astt toppart = mk_extract(result, (sz * 2) - 1, sz - 1);

      // Create a now base 2 type
      type2tc newtype = unsignedbv_type2tc(sz + 1);

      // All one bit vector is tricky, might be 64 bits wide for all we know.
      expr2tc allonesexpr = constant_int2tc(newtype, BigInt::power2m1(sz + 1));
      smt_astt allonesvector = convert_ast(allonesexpr);

      // It should either be zero or all one's;
      smt_astt all_ones = mk_eq(toppart, allonesvector);

      smt_astt all_zeros = mk_eq(toppart, convert_ast(gen_zero(newtype)));

      smt_astt lor = mk_or(all_ones, all_zeros);
      return mk_not(lor);
    }
    else if (int_encoding) // Handles both signed and unsigned cases
    {
      // Create a new integer type of size sz + 1
      type2tc newtype =
        is_signed ? signedbv_type2tc(sz + 1) : unsignedbv_type2tc(sz + 1);

      // Define upper bound for overflow detection
      expr2tc max_bound_expr;
      if (is_signed)
      {
        // Signed bounds: MIN_INT and MAX_INT
        expr2tc min_bound_expr =
          constant_int2tc(newtype, -BigInt::power2(sz - 1));
        max_bound_expr = constant_int2tc(newtype, BigInt::power2(sz - 1) - 1);

        smt_astt min_bound = convert_ast(min_bound_expr);
        smt_astt max_bound = convert_ast(max_bound_expr);

        // Convert result to signed type and check if it's out of bounds
        smt_astt overflow_high = mk_lt(result, min_bound);
        smt_astt overflow_low = mk_gt(result, max_bound);

        return mk_or(overflow_high, overflow_low);
      }
      else
      {
        // Unsigned upper bound (MAX_UINT)
        max_bound_expr = constant_int2tc(newtype, BigInt::power2(sz) - 1);

        smt_astt max_bound = convert_ast(max_bound_expr);

        // Overflow occurs if result > MAX_UINT
        smt_astt overflow_detected = mk_gt(result, max_bound);
        return overflow_detected;
      }
    }

    // Extract top half.
    smt_astt toppart = mk_extract(result, (sz * 2) - 1, sz);

    // It should be zero; if not, overflow
    smt_astt iseq = mk_eq(toppart, convert_ast(zero));
    return mk_not(iseq);
  }

  default:
    log_error("unexpected overflow_arith operand");
    abort();
  }

  return nullptr;
}

smt_astt smt_convt::overflow_cast(const expr2tc &expr)
{
  const overflow_cast2t &ocast = to_overflow_cast2t(expr);
  unsigned int src_width = ocast.operand->type->get_width();
  unsigned int dst_width = ocast.bits;

  // Validate the destination width
  if (dst_width == 0 || dst_width > src_width)
  {
    log_error(
      "SMT conversion: Invalid typecast width {} (source width: {})",
      dst_width,
      src_width);
    abort();
  }

  // Overflow is only relevant for signed/unsigned integers
  if (is_bool_type(ocast.operand->type))
    return mk_smt_bool(false);

  // Define unsigned bounds for the destination width
  expr2tc lower_bound = constant_int2tc(ocast.operand->type, 0);
  expr2tc upper_bound =
    constant_int2tc(ocast.operand->type, BigInt::power2(dst_width) - 1);

  // Check if the operand is within bounds
  expr2tc overflow_check = or2tc(
    lessthan2tc(ocast.operand, lower_bound),
    greaterthan2tc(ocast.operand, upper_bound));

  return convert_ast(overflow_check);
}

smt_astt smt_convt::overflow_neg(const expr2tc &expr)
{
  // Extract operand
  const overflow_neg2t &neg = to_overflow_neg2t(expr);
  unsigned int width = neg.operand->type->get_width();

  // Check if operand is unsigned
  bool is_unsigned = is_unsignedbv_type(neg.operand->type);

  if (is_unsigned)
  {
    // **Unsigned Negation Overflow Check**
    // In unsigned arithmetic, negation (-x) is effectively (UINT_MAX + 1 - x).
    // Any nonzero value negated will wrap around, which is unexpected behavior.
    expr2tc zero = constant_int2tc(neg.operand->type, 0);
    // Overflow occurs if operand is x != 0
    expr2tc val = notequal2tc(neg.operand, zero);

    return convert_ast(val);
  }
  else
  {
    // **Signed Negation Overflow Check**
    // Cast operand to signed type
    expr2tc operand = typecast2tc(signedbv_type2tc(width), neg.operand);
    simplify(operand);

    // Compute the minimum integer value for the operand's type (INT_MIN)
    expr2tc min_int =
      constant_int2tc(operand->type, -BigInt::power2(width - 1));

    // Overflow occurs if operand is INT_MIN
    expr2tc val = equality2tc(operand, min_int);

    return convert_ast(val);
  }
}
