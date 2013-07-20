#include "bitblast_conv.h"

bitblast_convt::bitblast_convt(bool enable_cache, bool int_encoding,
                               const namespacet &_ns, bool is_cpp,
                               bool tuple_support, bool bools_in_arrs,
                               bool can_init_inf_arrs)
  : smt_convt(enable_cache, int_encoding, _ns, is_cpp, tuple_support,
              bools_in_arrs, can_init_inf_arrs)
{
}

bitblast_convt::~bitblast_convt()
{
}

smt_ast *
bitblast_convt::mk_func_app(const smt_sort *ressort __attribute__((unused)),
                            smt_func_kind f __attribute__((unused)),
                            const smt_ast * const *args __attribute__((unused)),
                            unsigned int num __attribute__((unused)))
{
  abort();
}

void
bitblast_convt::eliminate_duplicates(const bvt &bv, bvt &dest)
{
  std::set<literalt> s;

  dest.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin(); it!=bv.end(); it++)
  {
    if(s.insert(*it).second)
      dest.push_back(*it);
  }
}

void
bitblast_convt::full_adder(const bvt &op0, const bvt &op1, bvt &output,
                          literalt carry_in, literalt &carry_out)
{
  assert(op0.size() == op1.size());
  output.reserve(op0.size());

  carry_out = carry_in;

  for (unsigned int i = 0; i < op0.size(); i++) {
    output.push_back(lxor(lxor(op0[i], op1[i]), carry_out));
    carry_out = carry(op0[i], op1[i], carry_out);
  }

  return;
}

literalt
bitblast_convt::carry(literalt a, literalt b, literalt c)
{
  bvt tmp;
  tmp.reserve(3);
  tmp.push_back(land(a, b));
  tmp.push_back(land(a, c));
  tmp.push_back(land(b, c));
  return lor(tmp);
}

literalt
bitblast_convt::unsigned_less_than(const bvt &arg0, const bvt &arg1)
{
  bvt tmp = arg1;
  invert(tmp);
  return lnot(carry_out(arg0, tmp, const_literal(true)));
}

void
bitblast_convt::unsigned_multiplier(const bvt &op0, const bvt &op1, bvt &output)
{
  output.resize(op0.size());

  for (unsigned int i = 0; i < op0.size(); i++)
    output[i] = const_literal(false);

  for (unsigned int i = 0; i < op0.size(); i++) {
    if (op0[i] != const_literal(false)) {
      bvt tmpop;
      tmpop.reserve(op0.size());

      for (unsigned int idx = 0; idx < i; idx++)
        tmpop.push_back(const_literal(false));

      for (unsigned int idx = i; idx < op0.size(); idx++)
        tmpop.push_back(land(op1[idx-i], op0[i]));

      bvt tmpadd;
      literalt dummy;
      full_adder(output, tmpop, tmpadd, const_literal(false), dummy);
      output = tmpadd;
    }
  }
}

void
bitblast_convt::signed_multiplier(const bvt &op0, const bvt &op1, bvt &output)
{
  assert(op0.size() == op1.size() && op0.size() != 0);
  literalt sign0 = op0[op0.size()-1];
  literalt sign1 = op1[op1.size()-1];

  bvt neg0, neg1;
  cond_negate(op0, neg0, sign0);
  cond_negate(op1, neg1, sign1);

  bvt tmp;
  unsigned_multiplier(neg0, neg1, tmp);

  literalt res_sign = lxor(sign0, sign1);

  cond_negate(tmp, output, res_sign);
}

void
bitblast_convt::cond_negate(const bvt &vals, bvt &out, literalt cond)
{
  bvt inv;
  negate(vals, inv);

  out.resize(vals.size());

  for (unsigned int i = 0; i < vals.size(); i++)
    out[i] = lselect(cond, inv[i], vals[i]);
  
  return;
}

void
bitblast_convt::negate(const bvt &inp, bvt &oup)
{
  oup.resize(inp.size());
  bvt inv = inp;
  invert(inv);

  literalt dummy;
  incrementer(inv, const_literal(true), dummy, oup);
  return;
}

void
bitblast_convt::incrementer(const bvt &inp, const literalt &carryin,
                           literalt carryout, bvt &oup)
{
  carryout = carryin;

  for (unsigned int i = 0; i < inp.size(); i++) {
    literalt new_carry = land(carryout, inp[i]);
    oup[i] = lxor(inp[i], carryout);
    carryout = new_carry;
  }

  return;
}

void
bitblast_convt::signed_divider(const bvt &op0, const bvt &op1, bvt &res,
                              bvt &rem)
{
  assert(op0.size() == op1.size());

  bvt _op0;
  bvt _op1;
  _op0.resize(op0.size());
  _op1.resize(op1.size());

  literalt sign0 = op0[op0.size() - 1];
  literalt sign1 = op1[op1.size() - 1];

  bvt neg0;
  bvt neg1;
  negate(op0, neg0);
  negate(op1, neg1);

  for (unsigned int i = 0; i < op0.size(); i++)
    _op0[i] = lselect(sign0, neg0[i], op0[i]);

  for (unsigned int i = 0; i < op1.size(); i++)
    _op1[i] = lselect(sign1, neg1[i], op1[i]);

  unsigned_divider(_op0, _op1, res, rem);

  bvt neg_res;
  bvt neg_rem;
  negate(res, neg_res);
  negate(rem, neg_rem);

  literalt result_sign = lxor(sign0, sign1);

  for (unsigned int i = 0; i < res.size(); i++)
    res[i] = lselect(result_sign, neg_res[i], res[i]);

  for (unsigned int i = 0; i < rem.size(); i++)
    rem[i] = lselect(result_sign, neg_rem[i], rem[i]);

  return;
}

void
bitblast_convt::unsigned_divider(const bvt &op0, const bvt &op1, bvt &res,
                                bvt &rem)
{
  assert(op0.size() == op1.size());
  unsigned int width = op0.size();
  res.resize(width);
  rem.resize(width);

  literalt is_not_zero = lor(op1);

  for (unsigned int i = 0; i < width; i++) {
    res[i] = new_variable();
    rem[i] = new_variable();
  }

  bvt product;
  unsigned_multiplier_no_overflow(res, op1, product);

  // "res*op1 + rem = op0"

  bvt sum;
  adder_no_overflow(product, rem, sum);

  literalt is_equal = equal(sum, op0);

  assert_lit(limplies(is_not_zero, is_equal));

  // "op1 != 0 => rem < op1"

  assert_lit(limplies(is_not_zero, lt_or_le(false, rem, op1, false)));

  // "op1 != 0 => res <= op0"

  assert_lit(limplies(is_not_zero, lt_or_le(true, rem, op0, false)));
}

void
bitblast_convt::unsigned_multiplier_no_overflow(const bvt &op0, const bvt &op1,
                                               bvt &res)
{
  assert(op0.size() == op1.size());
  bvt _op0 = op0, _op1 = op1;

  if (is_constant(_op1))
    std::swap(_op0, _op1);

  res.resize(op0.size());

  for (unsigned int i = 0; i < res.size(); i++)
    res[i] = const_literal(false);

  for (unsigned int sum = 0; sum < op0.size(); sum++) {
    if (op0[sum] != const_literal(false)) {
      bvt tmpop;

      tmpop.reserve(res.size());

      for (unsigned int idx = 0; idx < sum; idx++)
        tmpop.push_back(const_literal(false));

      for (unsigned int idx = sum; idx < res.size(); idx++)
        tmpop.push_back(land(op1[idx-sum], op0[sum]));

      bvt copy = res;
      adder_no_overflow(copy, tmpop, res);

      for (unsigned int idx = op1.size() - sum; idx < op1.size(); idx++) {
        literalt tmp = land(op1[idx], op0[sum]);
        tmp.invert();
        assert_lit(tmp);
      }
    }
  }
}

void
bitblast_convt::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res,
                                 bool subtract, bool is_signed)
{
  assert(op0.size() == op1.size());
  unsigned int width = op0.size();;
  bvt tmp_op1 = op1;
  if (subtract)
    negate(op1, tmp_op1);

  if (is_signed) {
    literalt old_sign = op0[width-1];
    literalt sign_the_same = lequal(op0[width-1], tmp_op1[width-1]);
    literalt carry;
    full_adder(op0, tmp_op1, res, const_literal(subtract), carry);
    literalt stop_overflow = land(sign_the_same, lxor(op0[width-1], old_sign));
    stop_overflow.invert();
    assert_lit(stop_overflow);
  } else {
    literalt carry_out;
    full_adder(op0, tmp_op1, res, const_literal(subtract), carry_out);
    if (subtract) {
      assert_lit(carry_out);
    } else {
      carry_out.invert();
      assert_lit(carry_out);
    }
  }

  return;
}

void
bitblast_convt::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res)
{
  res.resize(op0.size());

  literalt carry_out = const_literal(false);
  for (unsigned int i = 0; i < op0.size(); i++) {
    literalt op0_bit = op0[i];

    res[i] = lxor(lxor(op0_bit, op1[i]), carry_out);
    carry_out = carry(op0_bit, op1[i], carry_out);
  }

  carry_out.invert();
  assert_lit(carry_out);
}

bool
bitblast_convt::is_constant(const bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    if (!bv[i].is_constant())
      return false;
  return true;
}

literalt
bitblast_convt::carry_out(const bvt &a, const bvt &b, literalt c)
{
  literalt carry_out = c;

  for (unsigned int i = 0; i < a.size(); i++)
    carry_out = carry(a[i], b[i], carry_out);

  return carry_out;
}

literalt
bitblast_convt::equal(const bvt &op0, const bvt &op1)
{
  assert(op0.size() == op1.size());
  bvt tmp;
  tmp.reserve(op0.size());

  for (unsigned int i = 0; i < op0.size(); i++)
    tmp.push_back(lequal(op0[i], op1[i]));

  literalt res = land(tmp);
  return res;
}

literalt
bitblast_convt::lt_or_le(bool or_equal, const bvt &bv0, const bvt &bv1,
                        bool is_signed)
{
  assert(bv0.size() == bv1.size());
  literalt top0 = bv0[bv0.size() - 1], top1 = bv1[bv1.size() - 1];

  bvt inv_op1 = bv1;
  invert(inv_op1);
  literalt carry = carry_out(bv0, inv_op1, const_literal(true));

  literalt result;
  if (is_signed)
    result = lxor(lequal(top0, top1), carry);
  else
    result = lnot(carry);

  if (or_equal)
    result = lor(result, equal(bv0, bv1));

  return result;
}

void
bitblast_convt::invert(bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    bv[i] = lnot(bv[i]);
}

void
bitblast_convt::barrel_shift(const bvt &op, const shiftt s, const bvt &dist,
                            bvt &out)
{
  unsigned long d = 1;
  out = op;

  // FIXME: clip this on oversized shifts?
  for (unsigned int pos = 0; pos < dist.size(); pos++) {
    if (dist[pos] != const_literal(false)) {
      bvt tmp;
      shift(out, s, d, tmp);

      for (unsigned int i = 0; i < op.size(); i++)
        out[i] = lselect(dist[pos], tmp[i], out[i]);
    }

    d <<= 1;
  }
}

void
bitblast_convt::shift(const bvt &inp, const shiftt &s, unsigned long d, bvt &out)
{
  out.resize(inp.size());

  for (unsigned int i = 0; i < inp.size(); i++) {
    literalt l;

    switch (s) {
    case LEFT:
      l = ((d <= i) ? inp[i-d] : const_literal(false));
      break;
    case ARIGHT:
      l = ((i+d < inp.size()) ? inp[i+d] : inp[inp.size()-1]);
      break;
    case LRIGHT:
      l = ((i+d < inp.size()) ? inp[i+d] : const_literal(false));
      break;
    }

    out[i] = l;
  }

  return;
}

void
bitblast_convt::bvand(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(land(bv0[i], bv1[i]));

  return;
}

void
bitblast_convt::bvor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lor(bv0[i], bv1[i]));

  return;
}

void
bitblast_convt::bvxor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lxor(bv0[i], bv1[i]));

  return;
}

void
bitblast_convt::bvnot(const bvt &bv0, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lnot(bv0[i]));

  return;
}

literalt
bitblast_convt::land(const bvt &bv)
{
  if (bv.size() == 0)
    return const_literal(true);
  else if (bv.size() == 1)
    return bv[0];
  else if (bv.size() == 2)
    return land(bv[0], bv[1]);

  unsigned int trues = 0;
  for (unsigned int i = 0; i < bv.size(); i++) {
    if (bv[i] == const_literal(false))
      return const_literal(false);
    else if (bv[i] == const_literal(true))
      trues++;
  }

  if (trues == bv.size())
    return const_literal(true);

  bvt new_bv;

  eliminate_duplicates(bv, new_bv);

  literalt lit = new_variable();

  for (unsigned int i = 0; i < new_bv.size(); i++) {
    bvt lits;
    lits.reserve(2);
    lits.push_back(pos(new_bv[i]));
    lits.push_back(neg(lit));
    lcnf(lits);
  }

  bvt lits;
  lits.reserve(new_bv.size() + 1);

  for (unsigned int i = 0; i < new_bv.size(); i++)
    lits.push_back(neg(new_bv[i]));

  lits.push_back(pos(lit));
  lcnf(lits);

  return lit;
}

literalt
bitblast_convt::lor(const bvt &bv)
{
  if (bv.size() == 0) return const_literal(false);
  else if (bv.size() == 1) return bv[0];
  else if (bv.size() == 2) return lor(bv[0], bv[1]);

  for (unsigned int i = 0; i < bv.size(); i++)
    if (bv[i] == const_literal(true))
      return const_literal(true);

  bvt new_bv;
  eliminate_duplicates(bv, new_bv);

  literalt literal = new_variable();
  for (unsigned int i = 0; i < new_bv.size(); i++) {
    bvt lits;
    lits.reserve(2);
    lits.push_back(neg(new_bv[i]));
    lits.push_back(pos(literal));
    lcnf(lits);
  }

  bvt lits;
  lits.reserve(new_bv.size() + 1);

  for (unsigned int i = 0; i < new_bv.size(); i++)
    lits.push_back(pos(new_bv[i]));

  lits.push_back(neg(literal));
  lcnf(lits);

  return literal;
}
