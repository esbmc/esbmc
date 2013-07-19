#include <set>

#include <ansi-c/c_types.h>

#include "minisat_conv.h"

prop_convt *
create_new_minisat_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                          const optionst &options)
{
  return new minisat_convt(int_encoding, ns, is_cpp, options);
}

// Utility functions -- echoing CBMC quite a bit. The plan is to build up
// what's necessary, then doing all the required abstractions.

literalt
minisat_convt::new_variable()
{
  literalt l;
  Minisat::Var tmp = solver.newVar();
  l.set(tmp, false);
  no_variables = tmp+1;
  return l;
}

void
minisat_convt::convert(const bvt &bv, Minisat::vec<Lit> &dest)
{
  dest.capacity(bv.size());

  for (unsigned int i = 0; i < bv.size(); i++) {
    if (!bv[i].is_false())
      dest.push(Minisat::mkLit(bv[i].var_no(), bv[i].sign()));
  }
  return;
}

literalt
minisat_convt::lnot(literalt a)
{
  a.invert();
  return a;
}

literalt
minisat_convt::lselect(literalt a, literalt b, literalt c)
{  // a?b:c = (a AND b) OR (/a AND c)
  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;

  bvt bv;
  bv.reserve(2);
  bv.push_back(land(a, b));
  bv.push_back(land(lnot(a), c));
  return lor(bv);
}

literalt
minisat_convt::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

literalt
minisat_convt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
minisat_convt::lxor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return lnot(b);
  if (b == const_literal(true)) return lnot(a);

  literalt output = new_variable();
  gate_xor(a, b, output);
  return output;
}

literalt
minisat_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);

  literalt output = new_variable();
  gate_or(a, b, output);
  return output;
}

literalt
minisat_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  literalt output = new_variable();
  gate_and(a, b, output);
  return output;
}


literalt
minisat_convt::land(const bvt &bv)
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

void
minisat_convt::bvand(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(land(bv0[i], bv1[i]));

  return;
}

void
minisat_convt::bvor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lor(bv0[i], bv1[i]));

  return;
}

void
minisat_convt::bvxor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lxor(bv0[0], bv1[1]));

  return;
}

void
minisat_convt::bvnot(const bvt &bv0, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(lnot(bv0[i]));

  return;
}

literalt
minisat_convt::lor(const bvt &bv)
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


void
minisat_convt::gate_xor(literalt a, literalt b, literalt o)
{
  // a xor b = o <==> (a' + b' + o')
  //                  (a + b + o' )
  //                  (a' + b + o)
  //                  (a + b' + o)
  bvt lits;

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

void
minisat_convt::gate_or(literalt a, literalt b, literalt o)
{
  // a+b=c <==> (a' + c)( b' + c)(a + b + c')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);
}

void
minisat_convt::gate_and(literalt a, literalt b, literalt o)
{
  // a*b=c <==> (a + o')( b + o')(a'+b'+o)
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(a));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

void
minisat_convt::setto(literalt a, bool val)
{
  bvt b;
  if (val)
    b.push_back(a);
  else
    b.push_back(lnot(a));

  Minisat::vec<Lit> l;
  convert(b, l);
  solver.addClause_(l);
  return;
}

void
minisat_convt::set_equal(literalt a, literalt b)
{
  if (a == const_literal(false)) {
    setto(b, false);
    return;
  } else if (b == const_literal(false)) {
    setto(a, false);
    return;
  } else if (a == const_literal(true)) {
    setto(b, true);
    return;
  } else if (b == const_literal(true)) {
    setto(a, true);
    return;
  }

  bvt bv;
  bv.resize(2);
  bv[0] = a;
  bv[1] = lnot(b);
  Minisat::vec<Lit> l;
  convert(bv, l);
  solver.addClause_(l);

  bv[0] = lnot(a);
  bv[1] = b;
  l.clear();
  convert(bv, l);
  solver.addClause_(l);
  return;
}

void
minisat_convt::lcnf(const bvt &bv)
{
  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.empty())
    return;

  Minisat::vec<Lit> c;
  convert(bv, c);
  solver.addClause_(c);
  return;
}

void
minisat_convt::eliminate_duplicates(const bvt &bv, bvt &dest)
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
minisat_convt::full_adder(const bvt &op0, const bvt &op1, bvt &output,
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
minisat_convt::carry(literalt a, literalt b, literalt c)
{
  bvt tmp;
  tmp.reserve(3);
  tmp.push_back(land(a, b));
  tmp.push_back(land(a, c));
  tmp.push_back(land(b, c));
  return lor(tmp);
}

literalt
minisat_convt::unsigned_less_than(const bvt &arg0, const bvt &arg1)
{
  bvt tmp = arg1;
  invert(tmp);
  return lnot(carry_out(arg0, tmp, const_literal(true)));
}

void
minisat_convt::unsigned_multiplier(const bvt &op0, const bvt &op1, bvt &output)
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
minisat_convt::signed_multiplier(const bvt &op0, const bvt &op1, bvt &output)
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
minisat_convt::cond_negate(const bvt &vals, bvt &out, literalt cond)
{
  bvt inv;
  negate(vals, inv);

  out.resize(vals.size());

  for (unsigned int i = 0; i < vals.size(); i++)
    out[i] = lselect(cond, inv[i], vals[i]);
  
  return;
}

void
minisat_convt::negate(const bvt &inp, bvt &oup)
{
  oup.resize(inp.size());
  bvt inv = inp;
  invert(inv);

  literalt dummy;
  incrementer(inv, const_literal(true), dummy, oup);
  return;
}

void
minisat_convt::incrementer(const bvt &inp, const literalt &carryin,
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
minisat_convt::signed_divider(const bvt &op0, const bvt &op1, bvt &res,
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
minisat_convt::unsigned_divider(const bvt &op0, const bvt &op1, bvt &res,
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
minisat_convt::unsigned_multiplier_no_overflow(const bvt &op0, const bvt &op1,
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
minisat_convt::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res,
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
minisat_convt::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res)
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
minisat_convt::is_constant(const bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    if (!bv[i].is_constant())
      return false;
  return true;
}

literalt
minisat_convt::carry_out(const bvt &a, const bvt &b, literalt c)
{
  literalt carry_out = c;

  for (unsigned int i = 0; i < a.size(); i++)
    carry_out = carry(a[i], b[i], carry_out);

  return carry_out;
}

literalt
minisat_convt::equal(const bvt &op0, const bvt &op1)
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
minisat_convt::lt_or_le(bool or_equal, const bvt &bv0, const bvt &bv1,
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
minisat_convt::invert(bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    bv[i] = lnot(bv[i]);
}

void
minisat_convt::barrel_shift(const bvt &op, const shiftt s, const bvt &dist,
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
minisat_convt::shift(const bvt &inp, const shiftt &s, unsigned long d, bvt &out)
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

minisat_convt::minisat_convt(bool int_encoding, const namespacet &_ns,
                             bool is_cpp, const optionst &_opts)
         : smt_convt(true, int_encoding, _ns, is_cpp, false, true, true),
           array_indexes(), array_values(), array_updates(),
           solver(), options(_opts)
{
  smt_post_init();
}

minisat_convt::~minisat_convt(void)
{
}

prop_convt::resultt
minisat_convt::dec_solve()
{
  add_array_constraints();
  bool res = solver.solve();
  if (res)
    return prop_convt::P_SATISFIABLE;
  else
    return prop_convt::P_UNSATISFIABLE;
}

expr2tc
minisat_convt::get_bool(const smt_ast *a)
{
  const minisat_smt_ast *mast = minisat_ast_downcast(a);
  tvt t = l_get(mast->bv[0]);
  if (t.is_true())
    return true_expr;
  else if (t.is_false())
    return false_expr;
  else
    return expr2tc();
}

expr2tc
minisat_convt::get_bv(const type2tc &t, const smt_ast *a)
{
  const minisat_smt_ast *mast = minisat_ast_downcast(a);
  unsigned int sz = t->get_width();
  assert(sz <= 64 && "Your integers are larger than a uint64_t");
  assert(mast->bv.size() == sz);
  uint64_t accuml = 0;
  for (unsigned int i = 0; i < sz; i++) {
    uint64_t mask = 1 << i;
    tvt t = l_get(mast->bv[i]);
    if (t.is_true())
      accuml |= mask;
    else if (t.is_false())
      ; // It's zero
    else
      ; // It's undefined in this model. So may as well be zero.
  }

  return constant_int2tc(t, BigInt(accuml));
}

void
minisat_convt::dump_bv(const bvt &bv) const
{
  for (unsigned int i = 0; i < bv.size(); i++) {
    if (bv[i] == const_literal(false))
      std::cerr << "0";
    else if (bv[i] == const_literal(true))
      std::cerr << "1";
    else
      std::cerr << "?";
  }

  std::cerr << " " << bv.size() << std::endl;
  return;
}

expr2tc
minisat_convt::get(const expr2tc &expr)
{

  const smt_ast *value = convert_ast(expr);

  // It can however have various types. We only deal with bools and bitvectors;
  // hand everything else off to additional modelling code.
  switch (expr->type->type_id) {
  case type2t::bool_id:
  {
    return get_bool(value);
  }
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  {
    return get_bv(expr->type, value);
  }
  case type2t::fixedbv_id:
  {
    expr2tc tmp = get_bv(expr->type, value);
    const constant_int2t &intval = to_constant_int2t(tmp);
    uint64_t val = intval.constant_value.to_ulong();
    std::stringstream ss;
    ss << val;
    constant_exprt value_expr(migrate_type_back(expr->type));
    value_expr.set_value(get_fixed_point(expr->type->get_width(), ss.str()));
    fixedbvt fbv;
    fbv.from_expr(value_expr);
    return constant_fixedbv2tc(expr->type, fbv);
  }
  case type2t::array_id:
  {
    if (is_tuple_array_ast_type(expr->type))
      return tuple_array_get(expr);
    else
      return array_get(value, expr->type);
  }
  case type2t::pointer_id:
  case type2t::struct_id:
  case type2t::union_id:
    return tuple_get(expr);
  default:
    std::cerr << "Unrecognized type id " << expr->type->type_id << " in minisat"
              << " get" << std::endl;
    abort();
  }
}

const std::string
minisat_convt::solver_text()
{
  return "MiniSAT";
}

tvt
minisat_convt::l_get(literalt l)
{
  if (l == const_literal(true))
    return tvt(tvt::TV_TRUE);
  else if (l == const_literal(false))
    return tvt(tvt::TV_FALSE);

  Minisat::lbool val = solver.modelValue(Minisat::mkLit(l.var_no(), l.sign()));
  int v = Minisat::toInt(val);
  if (v == 0)
    return tvt(tvt::TV_TRUE);
  else if (v == 1)
    return tvt(tvt::TV_FALSE);
  else
    return tvt(tvt::TV_UNKNOWN);
}

void
minisat_convt::assert_lit(const literalt &l)
{
  Minisat::vec<Lit> c;
  c.push(Minisat::mkLit(l.var_no(), l.sign()));
  solver.addClause_(c);
  return;
}

smt_ast*
minisat_convt::mk_func_app(const smt_sort *ressort __attribute__((unused)),
    smt_func_kind f, const smt_ast* const* _args, unsigned int numargs)
{
  const minisat_smt_ast *args[4];
  minisat_smt_ast *result = NULL;
  unsigned int i;

  assert(numargs < 4 && "Too many arguments to minisat_convt::mk_func_app");
  for (i = 0; i < numargs; i++)
    args[i] = minisat_ast_downcast(_args[i]);

  switch (f) {
  case SMT_FUNC_EQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    result = mk_ast_equality(args[0], args[1], ressort);
    break;
  }
  case SMT_FUNC_NOTEQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    result = mk_ast_equality(args[0], args[1], ressort);
    result->bv[0] = lnot(result->bv[0]);
    break;
  }
  case SMT_FUNC_NOT:
  {
    literalt res = lnot(args[0]->bv[0]);
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_OR:
  {
    literalt res = lor(args[0]->bv[0], args[1]->bv[0]);
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(limplies(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_ITE:
  {
    if (ressort->id == SMT_SORT_ARRAY) {
      minisat_array_ast *res;
      res = array_ite(args[0], minisat_array_downcast(_args[1]),
                         minisat_array_downcast(_args[2]),
                         minisat_sort_downcast(ressort));
      result = (minisat_smt_ast*)res;
    } else {
      assert(args[1]->bv.size() == args[2]->bv.size());
      result = new minisat_smt_ast(ressort);
      for (unsigned int i = 0; i < args[1]->bv.size(); i++)
        result->bv.push_back(lselect(args[0]->bv[0], args[1]->bv[i],
                                     args[2]->bv[i]));
    }
    break;
  }
  case SMT_FUNC_AND:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(land(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_XOR:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lxor(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_BVADD:
  {
    literalt carry_in = const_literal(false);
    literalt carry_out;
    result = new minisat_smt_ast(ressort);
    full_adder(args[0]->bv, args[1]->bv, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVSUB:
  {
    literalt carry_in = const_literal(true);
    literalt carry_out;
    result = new minisat_smt_ast(ressort);
    bvt op1 = args[1]->bv;
    invert(op1);
    full_adder(args[0]->bv, op1, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVUGT:
  {
    // Same as LT flipped
    std::swap(args[0], args[1]);
    result = mk_func_app(ressort, SMT_FUNC_BVULT, args, 2);
    break;
  }
  case SMT_FUNC_BVUGTE:
  {
    // This is the negative of less-than
    result = mk_func_app(ressort, SMT_FUNC_BVULT, args, 2);
    result->bv[0].invert();
    break;
  }
  case SMT_FUNC_BVULT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(unsigned_less_than(args[0]->bv, args[1]->bv));
    break;
  }
  case SMT_FUNC_BVULTE:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, false));
    break;
  }
  case SMT_FUNC_BVSGTE:
  {
    // This is the negative of less-than
    result = mk_func_app(ressort, SMT_FUNC_BVSLT, args, 2);
    result->bv[0].invert();
    break;
  }
  case SMT_FUNC_BVSLTE:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVSGT:
  {
    // Same as LT flipped
    std::swap(args[0], args[1]);
    result = mk_func_app(ressort, SMT_FUNC_BVSLT, args, 2);
    break;
  }
  case SMT_FUNC_BVSLT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(false, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVMUL:
  {
    result = new minisat_smt_ast(ressort);
    const minisat_smt_sort *sort0 = minisat_sort_downcast(args[0]->sort);
    const minisat_smt_sort *sort1 = minisat_sort_downcast(args[1]->sort);
    if (sort0->sign || sort1->sign) {
      signed_multiplier(args[0]->bv, args[1]->bv, result->bv);
    } else {
      unsigned_multiplier(args[0]->bv, args[1]->bv, result->bv);
    }
    break;
  }
  case SMT_FUNC_CONCAT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.insert(result->bv.begin(), args[0]->bv.begin(),
                      args[0]->bv.end());
    result->bv.insert(result->bv.begin(), args[1]->bv.begin(),
                      args[1]->bv.end());
    break;
  }
  case SMT_FUNC_BVAND:
  {
    result = new minisat_smt_ast(ressort);
    bvand(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVXOR:
  {
    result = new minisat_smt_ast(ressort);
    bvxor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVOR:
  {
    result = new minisat_smt_ast(ressort);
    bvor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVNOT:
  {
    result = new minisat_smt_ast(ressort);
    bvnot(args[0]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVASHR:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, shiftt::ARIGHT, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVLSHR:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, shiftt::LRIGHT, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSHL:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, shiftt::LEFT, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSDIV:
  {
    bvt rem;
    result = new minisat_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVUDIV:
  {
    bvt rem;
    result = new minisat_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVSMOD:
  {
    bvt res;
    result = new minisat_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVUMOD:
  {
    bvt res;
    result = new minisat_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVNEG:
  {
    result = new minisat_smt_ast(ressort);
    negate(args[0]->bv, result->bv);
    break;
  }
  default:
    std::cerr << "Unimplemented SMT function \"" << smt_func_name_table[f]
              << "\" in minisat convt" << std::endl;
    abort();
  }

  return result;
}

const smt_ast *
minisat_convt::lit_to_ast(const literalt &l)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.push_back(l);
  return a;
}

smt_sort*
minisat_convt::mk_sort(smt_sort_kind k, ...)
{
  va_list ap;
  minisat_smt_sort *s = NULL, *dom, *range;
  unsigned long uint;
  int thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    std::cerr << "Can't make Int sorts in Minisat" << std::endl;
    abort();
  case SMT_SORT_REAL:
    std::cerr << "Can't make Real sorts in Minisat" << std::endl;
    abort();
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    s = new minisat_smt_sort(k, uint, (bool)thebool);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, minisat_smt_sort *); // Consider constness?
    range = va_arg(ap, minisat_smt_sort *);
    s = new minisat_smt_sort(k, dom->width, range->width);
    break;
  case SMT_SORT_BOOL:
    s = new minisat_smt_sort(k);
    break;
  default:
    std::cerr << "Unimplemented SMT sort " << k << " in Minisat conversion"
              << std::endl;
    abort();
  }

  return s;
}

literalt
minisat_convt::mk_lit(const smt_ast *val)
{
  const minisat_smt_ast *a = minisat_ast_downcast(val);
  assert(a->sort->id == SMT_SORT_BOOL);
  assert(a->bv.size() == 1);

  literalt l = new_variable();
  set_equal(l, a->bv[0]);
  return l;
}

smt_ast*
minisat_convt::mk_smt_int(const mp_integer &intval __attribute__((unused)), bool sign __attribute__((unused)))
{
  std::cerr << "Can't create integers in minisat solver" << std::endl;
  abort();
}

smt_ast*
minisat_convt::mk_smt_real(const std::string &value __attribute__((unused)))
{
  std::cerr << "Can't create reals in minisat solver" << std::endl;
  abort();
}

smt_ast*
minisat_convt::mk_smt_bvint(const mp_integer &intval, bool sign,
                            unsigned int w)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, w, sign);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.resize(w);
  int64_t u = intval.to_long();
  for (unsigned int i = 0; i < w; i++) {
    int64_t mask = (1ULL << i);
    bool val = u & mask;
    a->bv[i] = const_literal(val);
  }

  return a;
}

smt_ast*
minisat_convt::mk_smt_bool(bool boolval)
{
  literalt l = const_literal(boolval);

  smt_sort *s = mk_sort(SMT_SORT_BOOL);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.push_back(l);
  return a;
}

smt_ast*
minisat_convt::mk_smt_symbol(const std::string &name, const smt_sort *sort)
{
  // Like metasmt, minisat doesn't have a symbol table. So, build our own.
  symtable_type::const_iterator it = sym_table.find(name);
  if (it != sym_table.end())
    return minisat_ast_downcast(it->second);

  // Otherwise, we need to build this AST ourselves.
  minisat_smt_ast *a = new minisat_smt_ast(sort);
  minisat_smt_sort *s = minisat_sort_downcast(sort);
  switch (sort->id) {
  case SMT_SORT_BOOL:
  {
    literalt l = new_variable();
    a->bv.push_back(l);
    break;
  }
  case SMT_SORT_BV:
  {
    // Bunch of fresh variables
    for (unsigned int i = 0; i < s->width; i++)
      a->bv.push_back(new_variable());
    break;
  }
  case SMT_SORT_ARRAY:
  {
    a = fresh_array(s, name);
    break;
  }
  default:
    std::cerr << "Unimplemented symbol type " << sort->id
              << " in minisat symbol creation" << std::endl;
    abort();
  }

  sym_table.insert(symtable_type::value_type(name, a));
  return a;
}

smt_sort*
minisat_convt::mk_struct_sort(const type2tc &t __attribute__((unused)))
{
  abort();
}

smt_sort*
minisat_convt::mk_union_sort(const type2tc &t __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_extract(const smt_ast *src, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  const minisat_smt_ast *mast = minisat_ast_downcast(src);
  minisat_smt_ast *result = new minisat_smt_ast(s);
  for (unsigned int i = low; i <= high; i++)
    result->bv.push_back(mast->bv[i]);

  return result;
}

minisat_smt_ast *
minisat_convt::mk_ast_equality(const minisat_smt_ast *a,
                               const minisat_smt_ast *b,
                               const smt_sort *ressort)
{
  switch (a->sort->id) {
  case SMT_SORT_BOOL:
  {
    literalt res = lequal(a->bv[0], b->bv[0]);
    minisat_smt_ast *n = new minisat_smt_ast(a->sort);
    n->bv.push_back(res);
    return n;
  }
  case SMT_SORT_BV:
  {
    const minisat_smt_sort *ms = minisat_sort_downcast(a->sort);
    minisat_smt_ast *n = new minisat_smt_ast(ressort);
    n->bv.push_back(equal(a->bv, b->bv));
    return n;
  }
  case SMT_SORT_ARRAY:
  {
    std::cerr << "No direct array equality support in MiniSAT converter"
              << std::endl;
    abort();
  }
  default:
    std::cerr << "Invalid sort " << a->sort->id << " for equality in minisat"
              << std::endl;
    abort();
  }
}

const smt_ast *
minisat_convt::convert_array_equality(const expr2tc &a, const expr2tc &b)
{

  // Only support a scenario where the lhs (a) is a symbol.
  assert(is_symbol2t(a) && "Malformed minisat array equality");

  const minisat_array_ast *value;

  // We want everything to go through the expression cache. Except when creating
  // new arrays with either constant_array_of or constant_array.
  if (is_constant_expr(b)) {
    value = minisat_array_downcast(array_create(b));
  } else {
    value = minisat_array_downcast(convert_ast(b));
  }

  const symbol2t &sym = to_symbol2t(a);
  std::string symname = sym.get_symbol_name();

  sym_table[symname] = value;

  // Also pump that into the smt cache.
  smt_cache_entryt e;
  e.val = a;
  e.ast = value;
  e.level = ctx_level;
  smt_cache.insert(e);

  // Return a true value, because that assignment is always true.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  minisat_smt_ast *result = new minisat_smt_ast(boolsort);
  result->bv.push_back(const_literal(true));
  return result;
}

const smt_ast *
minisat_convt::fresh_array(const minisat_smt_sort *ms, const std::string &name)
{
  // No solver representation for this.
  unsigned long domain_width = ms->get_domain_width();
  unsigned long array_size = 1UL << domain_width;
  const smt_sort *range_sort = mk_sort(SMT_SORT_BV, ms->arrrange_width, false);

  minisat_array_ast *mast = new minisat_array_ast(ms);
  mast->symname = name;
  sym_table.insert(symtable_type::value_type(name, mast));

  if (mast->is_unbounded_array()) {
    // Don't attempt to initialize. Store the fact that we've allocated a
    // fresh new array.
    mast->base_array_id = array_indexes.size();
    mast->array_update_num = 0;
    std::set<expr2tc> tmp_set;
    array_indexes.push_back(tmp_set);

    std::vector<std::list<struct array_select> > tmp2;
    array_values.push_back(tmp2);

    std::list<struct array_select> tmp25;
    array_values[mast->base_array_id].push_back(tmp25);

    std::vector<struct array_with> tmp3;
    array_updates.push_back(tmp3);

    // Aimless piece of data, just to keep indexes in iarray_updates and
    // array_values in sync.
    struct array_with w;
    w.is_ite = false;
    w.idx = expr2tc();
    array_updates[mast->base_array_id].push_back(w);

    array_subtypes.push_back(ms->arrrange_width);

    return mast;
  }

  mast->array_fields.reserve(array_size);

  // Populate that array with a bunch of fresh bvs of the correct sort.
  unsigned long i;
  for (i = 0; i < array_size; i++) {
    const smt_ast *a = mk_fresh(range_sort, "minisat_fresh_array::");
    mast->array_fields.push_back(a);
  }

  return mast;
}

const smt_ast *
minisat_convt::mk_select(const expr2tc &array, const expr2tc &idx,
                         const smt_sort *ressort)
{
  assert(ressort->id != SMT_SORT_ARRAY);
  minisat_array_ast *ma = minisat_array_downcast(convert_ast(array));

  if (ma->is_unbounded_array())
    return mk_unbounded_select(ma, idx, ressort);

  assert(ma->array_fields.size() != 0);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      // Return a fresh value.
      return mk_fresh(ressort, "minisat_mk_select_badidx::");

    // Otherwise,
    return ma->array_fields[intval];
  }

  // What we have here is a nondeterministic index. Alas, compare with
  // everything.
  const smt_ast *fresh = mk_fresh(ressort, "minisat_mk_select::");
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *args[2], *idxargs[2], *impargs[2];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  args[0] = fresh;
  idxargs[0] = real_idx;

  for (unsigned long i = 0; i < ma->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    args[1] = ma->array_fields[i];
    const smt_ast *val_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2);

    impargs[0] = idx_eq;
    impargs[1] = val_eq;

    const smt_ast *res = mk_func_app(bool_sort, SMT_FUNC_IMPLIES, impargs, 2);
    assert_lit(minisat_ast_downcast(res)->bv[0]);
  }

  return fresh;
}

const smt_ast *
minisat_convt::mk_store(const expr2tc &array, const expr2tc &idx,
                        const expr2tc &value, const smt_sort *ressort)
{
  minisat_array_ast *ma = minisat_array_downcast(convert_ast(array));

  if (ma->is_unbounded_array())
    return mk_unbounded_store(ma, idx, convert_ast(value), ressort);

  assert(ma->array_fields.size() != 0);

  minisat_array_ast *mast =
    new minisat_array_ast(ressort, ma->array_fields);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      return convert_ast(array);

    // Otherwise,
    mast->array_fields[intval] = convert_ast(value);
    return mast;
  }

  // Oh dear. We need to update /all the fields/ :(
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *real_value = convert_ast(value);
  const smt_ast *iteargs[3], *idxargs[2], *impargs[2], *accuml_props[2];
  unsigned long dom_width = mast->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  idxargs[0] = real_idx;
  iteargs[1] = real_value;

  for (unsigned long i = 0; i < mast->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);

    iteargs[0] = idx_eq;
    iteargs[2] = mast->array_fields[i];

    const smt_ast *new_val =
      mk_func_app(iteargs[1]->sort, SMT_FUNC_ITE, iteargs, 3);
    mast->array_fields[i] = new_val;
  }

  return mast;
}

const smt_ast *
minisat_convt::mk_unbounded_select(const minisat_array_ast *ma,
                                   const expr2tc &real_idx,
                                   const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(real_idx);

  // Generate a new free variable
  smt_ast *a = mk_fresh(ressort, "mk_unbounded_select");

  struct array_select sel;
  sel.src_array_update_num = ma->array_update_num;
  sel.idx = real_idx;
  sel.val = a;
  // Record this index
  array_values[ma->base_array_id][ma->array_update_num].push_back(sel);

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  convert_ast(real_idx);

  return a;
}

const smt_ast *
minisat_convt::mk_unbounded_store(const minisat_array_ast *ma,
                                  const expr2tc &idx, const smt_ast *value,
                                  const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(idx);

  // More nuanced: allocate a new array representation.
  minisat_array_ast *newarr = new minisat_array_ast(ressort);
  newarr->base_array_id = ma->base_array_id;
  newarr->array_update_num = array_updates[ma->base_array_id].size();

  // Record update
  struct array_with w;
  w.is_ite = false;
  w.idx = idx;
  w.u.w.src_array_update_num = ma->array_update_num;
  w.u.w.val = value;
  array_updates[ma->base_array_id].push_back(w);

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  convert_ast(idx);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[ma->base_array_id].push_back(tmp);

  // Result is the new array id goo.
  return newarr;
}

const minisat_array_ast *
minisat_convt::array_ite(const minisat_smt_ast *cond,
                         const minisat_array_ast *true_arr,
                         const minisat_array_ast *false_arr,
                         const minisat_smt_sort *thesort)
{

  if (true_arr->is_unbounded_array())
    return unbounded_array_ite(cond, true_arr, false_arr, thesort);

  // For each element, make an ite.
  assert(true_arr->array_fields.size() != 0 &&
         true_arr->array_fields.size() == false_arr->array_fields.size());
  minisat_array_ast *mast = new minisat_array_ast(thesort);
  const smt_ast *args[3];
  args[0] = cond;
  unsigned long i;
  for (i = 0; i < true_arr->array_fields.size(); i++) {
    // One ite pls.
    args[1] = true_arr->array_fields[i];
    args[2] = false_arr->array_fields[i];
    const smt_ast *res = mk_func_app(args[1]->sort, SMT_FUNC_ITE, args, 3);
    mast->array_fields.push_back(minisat_ast_downcast(res));
  }

  return mast;
}

const minisat_array_ast *
minisat_convt::unbounded_array_ite(const minisat_smt_ast *cond,
                                   const minisat_array_ast *true_arr,
                                   const minisat_array_ast *false_arr,
                                   const minisat_smt_sort *thesort)
{
  // Precondition for a lot of goo: that the two arrays are the same, at
  // different points in time.
  assert(true_arr->base_array_id == false_arr->base_array_id &&
         "ITE between two arrays with different bases are unsupported");

  minisat_array_ast *newarr = new minisat_array_ast(thesort);
  newarr->base_array_id = true_arr->base_array_id;
  newarr->array_update_num = array_updates[true_arr->base_array_id].size();

  struct array_with w;
  w.is_ite = true;
  w.idx = expr2tc();
  w.u.i.src_array_update_true = true_arr->array_update_num;
  w.u.i.src_array_update_false = false_arr->array_update_num;
  w.u.i.cond = cond;
  array_updates[true_arr->base_array_id].push_back(w);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[true_arr->base_array_id].push_back(tmp);

  return newarr;
}

const smt_ast *
minisat_convt::convert_array_of(const expr2tc &init_val,
                                unsigned long domain_width)
{
  const smt_sort *dom_sort = mk_sort(SMT_SORT_BV, domain_width, false);
  const smt_sort *idx_sort = convert_sort(init_val->type);

  if (!int_encoding && is_bool_type(init_val) && no_bools_in_arrays)
    idx_sort = mk_sort(SMT_SORT_BV, 1, false);

  const minisat_smt_sort *arr_sort =
    minisat_sort_downcast(mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort));

  minisat_array_ast *mast = new minisat_array_ast(arr_sort);

  const smt_ast *init = convert_ast(init_val);
  if (!int_encoding && is_bool_type(init_val) && no_bools_in_arrays)
    init = make_bool_bit(init);

  if (arr_sort->is_unbounded_array()) {
    delete mast;
    std::string name = mk_fresh_name("array_of_unbounded::");
    mast = minisat_array_downcast(fresh_array(arr_sort, name));
    array_of_vals.insert(std::pair<unsigned, const smt_ast *>
                                  (mast->base_array_id, init));
  } else {
    unsigned long array_size = 1UL << domain_width;
    for (unsigned long i = 0; i < array_size; i++)
      mast->array_fields.push_back(init);
  }

  return mast;
}

expr2tc
minisat_convt::fixed_array_get(const smt_ast *a, const type2tc &type)
{
  const minisat_array_ast *mast = minisat_array_downcast(a);
  const array_type2t &arr = to_array_type(type);

  std::vector<expr2tc> fields;
  fields.reserve(mast->array_fields.size());
  for (unsigned int i = 0; i < mast->array_fields.size(); i++) {
    fields.push_back(get_bv(arr.subtype, mast->array_fields[i]));
  }

  constant_array2tc result(type, fields);
  return result;
}

expr2tc
minisat_convt::array_get(const smt_ast *a, const type2tc &type)
{
  const minisat_smt_sort *s = minisat_sort_downcast(convert_sort(type));
  if (!s->is_unbounded_array()) {
    return fixed_array_get(a, type);
  }

  const array_type2t &t = to_array_type(type);

  const minisat_array_ast *mast = minisat_array_downcast(a);

  if (mast->base_array_id >= array_valuation.size()) {
    // This is an array that was not previously converted, therefore doesn't
    // appear in the valuation table. Therefore, all its values are free.
    return expr2tc();
  }

  // Fetch all the indexes
  const std::set<expr2tc> &indexes = array_indexes[mast->base_array_id];

  std::map<expr2tc, unsigned> idx_map;
  for (std::set<expr2tc>::const_iterator it = indexes.begin();
       it != indexes.end(); it++)
    idx_map.insert(std::pair<expr2tc, unsigned>(*it, idx_map.size()));

  // Pick a set of array values.
  const std::vector<const smt_ast *> &solver_values =
    array_valuation[mast->base_array_id][mast->array_update_num];

  // Evaluate each index and each value.
  BigInt::ullong_t max_idx = 0;
  std::vector<std::pair<expr2tc, expr2tc> > values;
  values.resize(idx_map.size());
  for (std::map<expr2tc, unsigned>::const_iterator it = idx_map.begin();
       it != idx_map.end(); it++) {
    expr2tc idx = it->first;
    if (!is_constant_expr(idx))
      idx = get(idx);

    const smt_ast *this_value = solver_values[it->second];

    // Read the valuation. Guarenteed not to be an array or struct.
    assert((this_value->sort->id == SMT_SORT_BOOL ||
            this_value->sort->id == SMT_SORT_BV) &&
           "Unexpected sort in array field");

    // unimplemented
    expr2tc real_value;
    if (this_value->sort->id == SMT_SORT_BOOL)
      real_value = get_bool(this_value);
    else
      real_value = get_bv(t.subtype, this_value);

    values[it->second] = std::pair<expr2tc, expr2tc>(idx, real_value);

    // And record the largest index
    max_idx = std::max(max_idx,
                       to_constant_int2t(idx).constant_value.to_ulong());
  }

  // Work out the size of the array. If it's too large, clip it. Fill the
  // remaining elements with their values. This is lossy: if we want accuracy
  // in the future, then we need to find a way of returning sparse arrays
  // for potentially unbounded array sizes.
  if (max_idx > 1024)
    max_idx = 1024;

  type2tc arr_type(new array_type2t(t.subtype,
                                constant_int2tc(index_type2(), BigInt(max_idx)),
                                false));
  std::vector<expr2tc> array_values;
  array_values.resize(max_idx);
  for (unsigned int i = 0; i < values.size(); i++) {
    const constant_int2t &ref = to_constant_int2t(values[i].first);
    uint64_t this_idx = ref.constant_value.to_ulong();
    if (this_idx >= max_idx)
      continue;

    array_values[this_idx] = values[i].second;
  }

  return constant_array2tc(arr_type, array_values);
}

void
minisat_convt::add_array_constraints(void)
{

  for (unsigned int i = 0; i < array_indexes.size(); i++) {
    add_array_constraints(i);
  }

  return;
}

void
minisat_convt::add_array_constraints(unsigned int arr)
{
  // Right: we need to tie things up regarding these bitvectors. We have a
  // set of indexes...
  const std::set<expr2tc> &indexes = array_indexes[arr];

  // What we're going to build is a two-dimensional vector ish of each element
  // at each point in time. Expensive, but meh.
  array_valuation.resize(array_valuation.size() + 1);
  std::vector<std::vector<const smt_ast *> > &real_array_values =
    array_valuation.back();

  // Subtype is thus
  const smt_sort *subtype = mk_sort(SMT_SORT_BV, array_subtypes[arr], false);

  // Pre-allocate all the storage.
  real_array_values.resize(array_values[arr].size());
  for (unsigned int i = 0; i < real_array_values.size(); i++)
    real_array_values[i].resize(indexes.size());

  // Compute a mapping between indexes and an element in the vector. These
  // are ordered by how std::set orders them, not by history or anything. Or
  // even the element index.
  std::map<expr2tc, unsigned> idx_map;
  for (std::set<expr2tc>::const_iterator it = indexes.begin();
       it != indexes.end(); it++)
    idx_map.insert(std::pair<expr2tc, unsigned>(*it, idx_map.size()));

  assert(idx_map.size() == indexes.size());

  // Initialize the first set of elements.
  std::map<unsigned, const smt_ast*>::const_iterator it =
    array_of_vals.find(arr);
  if (it != array_of_vals.end()) {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype, it->second);
  } else {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype);
  }

  add_initial_ackerman_constraints(real_array_values[0], idx_map, subtype);

  // Now repeatedly execute transitions between states.
  for (unsigned int i = 0; i < real_array_values.size() - 1; i++)
    execute_array_trans(real_array_values, arr, i, idx_map, subtype);

}

void
minisat_convt::execute_array_trans(
                            std::vector<std::vector<const smt_ast *> > &data,
                                   unsigned int arr,
                                   unsigned int idx,
                                   const std::map<expr2tc, unsigned> &idx_map,
                                   const smt_sort *subtype)
{
  // Steps: First, fill the destination vector with either free variables, or
  // the free variables that resulted for selects corresponding to that item.
  // Then apply update or ITE constraints.
  // Then apply equalities between the old and new values.

  std::vector<const smt_ast *> &dest_data = data[idx+1];
  collate_array_values(dest_data, idx_map, array_values[arr][idx+1], subtype);

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Two updates that could have occurred for this array: a simple with, or
  // an ite.
  const array_with &w = array_updates[arr][idx+1];
  if (w.is_ite) {
    // Turn every index element into an ITE representing this operation. Every
    // single element is addressed and updated; no further constraints are
    // needed. Not even the ackerman ones, in fact, because instances of that
    // from previous array updates will feed through to here (speculation).

    unsigned int true_idx = w.u.i.src_array_update_true;
    unsigned int false_idx = w.u.i.src_array_update_false;
    assert(true_idx < idx + 1 && false_idx < idx + 1);
    const std::vector<const smt_ast *> &true_vals = data[true_idx];
    const std::vector<const smt_ast *> &false_vals = data[false_idx];
    const smt_ast *cond = w.u.i.cond;

    // Each index value becomes an ITE between each source value.
    const smt_ast *args[3], *eq[2];
    args[0] = cond;
    for (unsigned int i = 0; i < idx_map.size(); i++) {
      args[1] = true_vals[i];
      args[2] = false_vals[i];
      eq[0] = mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
      eq[1] = dest_data[i];
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, eq, 2)));
    }
  } else {
    // Place a constraint on the updated variable; add equality constraints
    // between the older version and this version.

    // So, the updated element,
    std::map<expr2tc, unsigned>::const_iterator it = idx_map.find(w.idx);
    assert(it != idx_map.end());

    const expr2tc &update_idx_expr = it->first;
    const smt_ast *update_idx_ast = convert_ast(update_idx_expr);
    unsigned int updated_idx = it->second;
    const smt_ast *updated_value = w.u.w.val;

    // Assign in its value.
    dest_data[updated_idx] = updated_value;

    // Check all the values selected out of this instance; if any have the same
    // index, tie the select's fresh variable to the updated value. If there are
    // differing index exprs that evaluate to the same location they'll be
    // caught by code later.
    const std::list<struct array_select> &sels = array_values[arr][idx+1];
    for (std::list<struct array_select>::const_iterator it = sels.begin();
         it != sels.end(); it++) {
      if (it->idx == update_idx_expr) {
        const smt_ast *args[2];
        args[0] = updated_value;
        args[1] = it->val;
        assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }

    // Now look at all those other indexes...
    assert(w.u.w.src_array_update_num < idx+1);
    const std::vector<const smt_ast *> &source_data =
      data[w.u.w.src_array_update_num];

    const smt_ast *args[3];
    unsigned int i = 0;
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++, i++) {
      if (it2->second == updated_idx)
        continue;

      // Generate an ITE. If the index is nondeterministically equal to the
      // current index, take the updated value, otherwise the original value.
      // This departs from the CBMC implementation, in that they explicitly
      // use implies and ackerman constraints.
      // FIXME: benchmark the two approaches. For now, this is shorter.
      args[0] = update_idx_ast;
      args[1] = convert_ast(it2->first);
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      args[1] = updated_value;
      args[2] = source_data[i];
      args[0] = mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
      args[1] = dest_data[i];
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      // The latter part of this could be replaced with more complex logic,
      // that only asserts an equality between selected values, and just stores
      // the result of the ITE for all other values. FIXME: try this.
    }
  }
}

void
minisat_convt::collate_array_values(std::vector<const smt_ast *> &vals,
                                    const std::map<expr2tc, unsigned> &idx_map,
                                    const std::list<struct array_select> &idxs,
                                    const smt_sort *subtype,
                                    const smt_ast *init_val)
{
  // So, the value vector should be allocated but not initialized,
  assert(vals.size() == idx_map.size());

  // First, make everything null,
  for (std::vector<const smt_ast *>::iterator it = vals.begin();
       it != vals.end(); it++)
    *it = NULL;

  // Now assign in all free variables created as a result of selects.
  for (std::list<struct array_select>::const_iterator it = idxs.begin();
       it != idxs.end(); it++) {
    std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.find(it->idx);
    assert(it2 != idx_map.end());
    vals[it2->second] = it->val;
  }

  // Initialize everything else to either a free variable or the initial value.
  if (init_val == NULL) {
    // Free variables, except where free variables tied to selects have occurred
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL)
        *it = mk_fresh(subtype, "collate_array_vals::");
    }
  } else {
    // We need to assign the initial value in, except where there's already
    // a select/index, in which case we assert that the values are equal.
    const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL) {
        *it = init_val;
      } else {
        const smt_ast *args[2];
        args[0] = *it;
        args[1] = init_val;
        assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }
  }

  // Fin.
}

void
minisat_convt::add_initial_ackerman_constraints(
                                  const std::vector<const smt_ast *> &vals,
                                  const std::map<expr2tc,unsigned> &idx_map,
                                  const smt_sort *subtype)
{
  // Lolquadratic,
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  for (std::map<expr2tc, unsigned>::const_iterator it = idx_map.begin();
       it != idx_map.end(); it++) {
    const smt_ast *outer_idx = convert_ast(it->first);
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++) {
      const smt_ast *inner_idx = convert_ast(it2->first);

      // If they're the same idx, they're the same value.
      const smt_ast *args[2];
      args[0] = outer_idx;
      args[1] = inner_idx;
      const smt_ast *idxeq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = vals[it->second];
      args[1] = vals[it2->second];
      const smt_ast *valeq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = idxeq;
      args[1] = valeq;
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_IMPLIES, args, 2)));
    }
  }
}
