#include <solvers/smt/smt_conv.h>

smt_astt
smt_convt::overflow_arith(const expr2tc &expr)
{
  // If in integer mode, this is completely pointless. Return false.
  if (int_encoding)
    return mk_smt_bool(false);

  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);

  expr2tc zero = gen_zero(opers.side_1->type);
  lessthan2tc op1neg(opers.side_1, zero);
  lessthan2tc op2neg(opers.side_2, zero);

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed = (is_signedbv_type(opers.side_1) ||
                    is_signedbv_type(opers.side_2));

  switch (overflow.operand->expr_id)
  {
    case expr2t::add_id:
    {
      if (is_signed)
      {
        // Two cases: pos/pos, and neg/neg, which can over and underflow resp.
        // In pos/neg cases, no overflow or underflow is possible, for any value.
        lessthan2tc op1pos(zero, opers.side_1);
        lessthan2tc op2pos(zero, opers.side_2);
        and2tc both_pos(op1pos, op2pos);

        not2tc negop1(op1pos);
        not2tc negop2(op2pos);
        and2tc both_neg(negop1, negop2);

        implies2tc nooverflow(both_pos,
                              greaterthanequal2tc(overflow.operand, zero));
        implies2tc nounderflow(both_neg,
                              lessthanequal2tc(overflow.operand, zero));
        return convert_ast(not2tc(and2tc(nooverflow, nounderflow)));
      }

      // Just ensure the result is >= both operands.
      greaterthanequal2tc ge1(overflow.operand, opers.side_1);
      greaterthanequal2tc ge2(overflow.operand, opers.side_2);
      and2tc res(ge1, ge2);
      not2tc inv(res);
      return convert_ast(inv);
    }

    case expr2t::sub_id:
    {
      if (is_signed)
      {
        // Convert to be an addition
        neg2tc negop2(opers.side_2->type, opers.side_2);
        add2tc anadd(opers.side_1->type, opers.side_1, negop2);
        expr2tc add_overflows(new overflow2t(anadd));

        // Corner case: subtracting MIN_INT from many things overflows. The result
        // should always be positive.
        uint64_t topbit = 1ULL << (opers.side_1->type->get_width() - 1);
        constant_int2tc min_int(opers.side_1->type, BigInt(topbit));
        equality2tc is_min_int(min_int, opers.side_2);
        return convert_ast(or2tc(add_overflows, is_min_int));
      }

      // Just ensure the result is >= the operands.
      lessthanequal2tc le1(overflow.operand, opers.side_1);
      lessthanequal2tc le2(overflow.operand, opers.side_2);
      and2tc res(le1, le2);
      not2tc inv(res);
      return convert_ast(inv);
    }

    case expr2t::div_id:
    case expr2t::modulus_id:
    {
      if(is_signed)
      {
        // We can't divide -MIN_INT/-1
        uint64_t topbit = 1ULL << (opers.side_1->type->get_width() - 1);
        constant_int2tc min_int(opers.side_1->type, -BigInt(topbit));
        equality2tc is_min_int(min_int, opers.side_1);
        implies2tc imp(is_min_int, greaterthan2tc(overflow.operand, zero));

        constant_int2tc minus_one(opers.side_1->type, -BigInt(1));
        equality2tc is_minus_one(minus_one, opers.side_2);

        return convert_ast(and2tc(is_minus_one, is_min_int));
      }

      // No overflow for unsigned div/modulus?
      return nullptr;
    }

    case expr2t::shl_id:
    case expr2t::mul_id:
    {
      // Zero extend; multiply; Make a decision based on the top half.
      unsigned int sz = zero->type->get_width();

      smt_sortt normalsort = mk_sort(SMT_SORT_UBV, sz);
      smt_sortt bigsort = mk_sort(SMT_SORT_UBV, sz * 2);

      smt_astt arg1_ext = convert_ast(opers.side_1);
      arg1_ext = is_signedbv_type(opers.side_1) ?
        convert_sign_ext(arg1_ext, bigsort, sz, sz) :
        convert_zero_ext(arg1_ext, bigsort, sz);

      smt_astt arg2_ext = convert_ast(opers.side_2);
      arg2_ext = is_signedbv_type(opers.side_2) ?
        convert_sign_ext(arg2_ext, bigsort, sz, sz) :
        convert_zero_ext(arg2_ext, bigsort, sz);

      smt_astt result =
        mk_func_app(
          bigsort,
          (is_mul2t(overflow.operand) ? SMT_FUNC_BVMUL : SMT_FUNC_BVSHL),
          arg1_ext,
          arg2_ext);

      if (is_signed)
      {
        // Extract top half plus one (for the sign)
        smt_astt toppart = mk_extract(result, (sz * 2) - 1, sz-1, normalsort);

        // Create a now base 2 type
        unsignedbv_type2tc newtype(sz+1);

        // All one bit vector is tricky, might be 64 bits wide for all we know.
        constant_int2tc allonesexpr(newtype, BigInt((1ULL << (sz+1)) - 1));
        smt_astt allonesvector = convert_ast(allonesexpr);

        // It should either be zero or all one's;
        smt_astt all_ones =
          mk_func_app(boolean_sort, SMT_FUNC_EQ, toppart, allonesvector);

        smt_astt all_zeros =
          mk_func_app(boolean_sort, SMT_FUNC_EQ, toppart, convert_ast(gen_zero(newtype)));

        smt_astt lor =
          mk_func_app(boolean_sort, SMT_FUNC_OR, all_ones, all_zeros);
        return mk_func_app(boolean_sort, SMT_FUNC_NOT, &lor, 1);
      }

      // Extract top half.
      smt_astt toppart = mk_extract(result, (sz * 2) - 1, sz, normalsort);

      // It should be zero; if not, overflow
      smt_astt iseq =
        mk_func_app(boolean_sort, SMT_FUNC_EQ, toppart, convert_ast(zero));
      return mk_func_app(boolean_sort, SMT_FUNC_NOT, &iseq, 1);
    }

    default:
      std::cerr << "unexpected overflow_arith operand\n";
      abort();
  }

  return nullptr;
}

smt_astt
smt_convt::overflow_cast(const expr2tc &expr)
{
  // If in integer mode, this is completely pointless. Return false.
  if (int_encoding)
    return mk_smt_bool(false);

  const overflow_cast2t &ocast = to_overflow_cast2t(expr);
  unsigned int width = ocast.operand->type->get_width();
  unsigned int bits = ocast.bits;

  if (ocast.bits >= width || ocast.bits == 0) {
    std::cerr << "SMT conversion: overflow-typecast got wrong number of bits"
              << std::endl;
    abort();
  }

  // Basically: if it's positive in the first place, ensure all the top bits
  // are zero. If neg, then all the top are 1's /and/ the next bit, so that
  // it's considered negative in the next interpretation.

  constant_int2tc zero(ocast.operand->type, BigInt(0));
  lessthan2tc isnegexpr(ocast.operand, zero);
  smt_astt isneg = convert_ast(isnegexpr);
  smt_astt orig_val = convert_ast(ocast.operand);

  // Difference bits
  unsigned int pos_zero_bits = width - bits;
  unsigned int neg_one_bits = (width - bits) + 1;

  smt_sortt pos_zero_bits_sort =
    mk_sort(SMT_SORT_UBV, pos_zero_bits);
  smt_sortt neg_one_bits_sort =
    mk_sort(SMT_SORT_UBV, neg_one_bits);

  smt_astt pos_bits = mk_smt_bvint(BigInt(0), false, pos_zero_bits);
  smt_astt neg_bits = mk_smt_bvint(BigInt((1 << neg_one_bits) - 1),
                                         false, neg_one_bits);

  smt_astt pos_sel = mk_extract(orig_val, width - 1,
                                      width - pos_zero_bits,
                                      pos_zero_bits_sort);
  smt_astt neg_sel = mk_extract(orig_val, width - 1,
                                      width - neg_one_bits,
                                      neg_one_bits_sort);

  smt_astt pos_eq = mk_func_app(boolean_sort, SMT_FUNC_EQ, pos_bits, pos_sel);
  smt_astt neg_eq = mk_func_app(boolean_sort, SMT_FUNC_EQ, neg_bits, neg_sel);

  // isneg -> neg_eq, !isneg -> pos_eq
  smt_astt notisneg = mk_func_app(boolean_sort, SMT_FUNC_NOT, &isneg, 1);
  smt_astt c1 = mk_func_app(boolean_sort, SMT_FUNC_IMPLIES, isneg, neg_eq);
  smt_astt c2 = mk_func_app(boolean_sort, SMT_FUNC_IMPLIES, notisneg, pos_eq);

  smt_astt nooverflow = mk_func_app(boolean_sort, SMT_FUNC_AND, c1, c2);
  return mk_func_app(boolean_sort, SMT_FUNC_NOT, &nooverflow, 1);
}

smt_astt
smt_convt::overflow_neg(const expr2tc &expr)
{
  // If in integer mode, this is completely pointless. Return false.
  if (int_encoding)
    return mk_smt_bool(false);

  // Single failure mode: MIN_INT can't be neg'd
  const overflow_neg2t &neg = to_overflow_neg2t(expr);
  unsigned int width = neg.operand->type->get_width();

  constant_int2tc min_int(neg.operand->type, BigInt(1 << (width - 1)));
  equality2tc val(neg.operand, min_int);
  return convert_ast(val);
}
