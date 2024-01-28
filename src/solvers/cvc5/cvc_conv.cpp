#include <cstdint>
#include <util/c_types.h>
#include <cvc_conv.h>

#define new_ast new_solver_ast<cvc_smt_ast>

smt_convt *create_new_cvc_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api,
  fp_convt **fp_api)
{
  cvc_convt *conv = new cvc_convt(ns, options);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

cvc_convt::cvc_convt(const namespacet &ns, const optionst &options)
  : smt_convt(ns, options),
    array_iface(false, false),
    fp_convt(this),
    to_bv_counter(0),
    slv()
{
  // Already initialized stuff in the constructor list,
  slv.setOption("produce-models", "true");
  slv.setOption("produce-assertions", "true");
}

smt_convt::resultt cvc_convt::dec_solve()
{
  pre_solve();

  cvc5::Result r = slv.checkSat();
  if (r.isSat())
    return P_SATISFIABLE;

  if (r.isUnknown())
    return P_ERROR;

  return P_UNSATISFIABLE;
}

bool cvc_convt::get_bool(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);
  return e.getBooleanValue();
}

ieee_floatt cvc_convt::get_fpbv(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);
  //cvc5::FloatingPoint foo = e.getFloatingPointValue();

  ieee_floatt number(ieee_float_spect(
    // in mk_bvfp_sort() we added +1 for the sign bit
    a->sort->get_significand_width() - 1,
    a->sort->get_exponent_width()));
  return number;
#if 0 
  if (foo.isNaN())
    number.make_NaN();
  else if (foo.isInfinite())
  {
    if (foo.isPositive())
      number.make_plus_infinity();
    else
      number.make_minus_infinity();
  }
  else
    number.unpack(BigInt(foo.pack().toInteger().getUnsignedLong()));

  return number;
#endif
}

BigInt cvc_convt::get_bv(smt_astt a, bool is_signed)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);
  std::string str = e.getBitVectorValue();
  return binary2integer(str, is_signed);
}

expr2tc cvc_convt::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  auto const *carray = to_solver_smt_ast<cvc_smt_ast>(array);
  size_t orig_w = array->sort->get_domain_width();

  smt_astt tmpast = mk_smt_bv(BigInt(index), mk_bv_sort(orig_w));
  auto const *tmpa = to_solver_smt_ast<cvc_smt_ast>(tmpast);
  cvc5::Term e = slv.mkTerm(cvc5::Kind::SELECT, {carray->a, tmpa->a});
  delete tmpast;

  return get_by_ast(subtype, new_ast(e, convert_sort(subtype)));
}

const std::string cvc_convt::solver_text()
{
  std::stringstream ss;
  ss << "CVC " << slv.getVersion();
  return ss.str();
}

smt_astt cvc_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::ADD,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::SUB,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::MULT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_mod(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::INTS_MODULUS,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_div(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::INTS_DIVISION,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_shl(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);

  cvc5::Term p = slv.mkTerm(
    cvc5::Kind::POW,
    {to_solver_smt_ast<cvc_smt_ast>(mk_smt_bv(2, b->sort))->a,
     to_solver_smt_ast<cvc_smt_ast>(b)->a});
  return new_ast(
    slv.mkTerm(cvc5::Kind::MULT, {to_solver_smt_ast<cvc_smt_ast>(a)->a, p}),
    a->sort);
}

smt_astt cvc_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::NEG, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::LT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::GT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::LEQ,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::GEQ,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_real2int(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::TO_INTEGER, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_int2real(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT);
  return new_ast(
    slv.mkTerm(cvc5::Kind::TO_REAL, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_isint(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::IS_INTEGER, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt
cvc_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_FMA,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(v1)->a,
       to_solver_smt_ast<cvc_smt_ast>(v2)->a,
       to_solver_smt_ast<cvc_smt_ast>(v3)->a}),
    v1->sort);
}

smt_astt
cvc_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width)
{
  assert(width <= UINT32_MAX);
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const cvc_smt_ast *mrm =
    to_solver_smt_ast<cvc_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const cvc_smt_ast *mfrom = to_solver_smt_ast<cvc_smt_ast>(from);

  // AYB TODO, width is too large
  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_UBV, {width}); 

  return new_ast(
    slv.mkTerm(
      op,
      {mrm->a,
       mfrom->a}),
    mk_bv_sort(width));
}

smt_astt
cvc_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width)
{
  assert(width <= UINT32_MAX);
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const cvc_smt_ast *mrm =
    to_solver_smt_ast<cvc_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const cvc_smt_ast *mfrom = to_solver_smt_ast<cvc_smt_ast>(from);

  // AYB TODO, width is too large
  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_SBV, {width}); 

  return new_ast(
    slv.mkTerm(
      op,
      {mrm->a,
       mfrom->a}),
    mk_bv_sort(width));
}

smt_astt cvc_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_FP, {ew, sw});

  // AYB no idea here
  return new_ast(
    slv.mkTerm(
      op,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(from)->a}),
    to);
}

smt_astt
cvc_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_UBV, {ew, sw});

  return new_ast(
    slv.mkTerm(
      op,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(from)->a}),
    to);
}

smt_astt
cvc_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_SBV, {ew, sw});

  return new_ast(
    slv.mkTerm(
      op,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(from)->a}),
    to);
}

smt_astt cvc_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op2 = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {ew, sw});

  return new_ast(slv.mkTerm(op2, {to_solver_smt_ast<cvc_smt_ast>(op)->a}), to);
}

smt_astt cvc_convt::mk_from_fp_to_bv(smt_astt op)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(op);

  // Force NaN to always generate the same bv, otherwise, create a new variable
  const std::string name =
    (ca->a.getKind() == cvc5::Kind::CONST_FLOATINGPOINT &&
     ca->a.isFloatingPointNaN())
      ? "__ESBMC_NaN"
      : "__ESBMC_to_ieeebv" + std::to_string(to_bv_counter++);

  smt_astt new_symbol =
    mk_smt_symbol(name, mk_bv_sort(op->sort->get_data_width()));

  // and constraint it to be the conversion of the fp, since
  // (fp_matches_bv f bv) <-> (= f ((_ to_fp E S) bv))
  assert_ast(mk_eq(op, mk_from_bv_to_fp(new_symbol, op->sort)));

  return new_symbol;
}

smt_astt cvc_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_ADD,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_SUB,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_MULT,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_DIV,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_RTI,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(from)->a}),
    from->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_SQRT,
      {to_solver_smt_ast<cvc_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc_smt_ast>(rd)->a}),
    rd->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_EQ,
      {to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_GT,
      {to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_LT,
      {to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_GEQ,
      {to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_LEQ,
      {to_solver_smt_ast<cvc_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NAN,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_INF,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NORMAL,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_ZERO,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NEG,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_POS,
      {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_ABS, {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    op->sort);
}

smt_astt cvc_convt::mk_smt_fpbv_neg(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_NEG, {to_solver_smt_ast<cvc_smt_ast>(op)->a}),
    op->sort);
}

void cvc_convt::assert_ast(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);
  slv.assertFormula(ca->a);
}

smt_astt cvc_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ADD,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SUB,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_MULT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SREM,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UREM,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SDIV,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UDIV,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SHL,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ASHR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_LSHR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NEG, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NOT, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_XNOR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NOR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NAND,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_XOR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_OR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_AND,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::IMPLIES,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::XOR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::OR,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::AND,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::NOT, {to_solver_smt_ast<cvc_smt_ast>(a)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ULT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SLT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UGT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SGT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ULE,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SLE,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UGE,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SGE,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::EQUAL,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::DISTINCT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::STORE,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a,
       to_solver_smt_ast<cvc_smt_ast>(c)->a}),
    a->sort);
}

smt_astt cvc_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::SELECT,
      {to_solver_smt_ast<cvc_smt_ast>(a)->a,
       to_solver_smt_ast<cvc_smt_ast>(b)->a}),
    a->sort->get_range_sort());
}

smt_astt cvc_convt::mk_smt_int(const BigInt &theint)
{
  cvc5::Term e = slv.mkInteger(theint.to_int64());
  return new_ast(e, mk_int_sort());
}

smt_astt cvc_convt::mk_smt_real(const std::string &str)
{
  smt_sortt s = mk_real_sort();
  return new_ast(slv.mkReal(str), s);
}

smt_astt cvc_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  assert(thereal.spec.width() <= 64);
  smt_sortt s = mk_real_fp_sort(thereal.spec.e, thereal.spec.f);


  cvc5::Term float_bv = slv.mkBitVector(thereal.spec.width(), thereal.pack().to_uint64());
  return new_ast(
    slv.mkFloatingPoint(
      s->get_exponent_width(), s->get_significand_width(), float_bv), s);
}

smt_astt cvc_convt::mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw - 1);
  smt_astt the_nan = new_ast(
    slv.mkFloatingPointNaN(s->get_exponent_width(), s->get_significand_width()),
    s);

  if (sgn)
    the_nan = fp_convt::mk_smt_fpbv_neg(the_nan);

  return the_nan;
}

smt_astt cvc_convt::mk_smt_fpbv_inf(bool sign, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw - 1);

  // TODO: some template magic?
  cvc5::Term t = sign ? slv.mkFloatingPointNegInf(
                          s->get_exponent_width(), s->get_significand_width())
                      : slv.mkFloatingPointPosInf(
						  s->get_exponent_width(), s->get_significand_width());

  return new_ast(t, s);
}

smt_astt cvc_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  smt_sortt s = mk_fpbv_rm_sort();

  switch(rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    return new_ast(slv.mkRoundingMode(cvc5::RoundingMode::ROUND_NEAREST_TIES_TO_EVEN), s);
  case ieee_floatt::ROUND_TO_MINUS_INF:
    return new_ast(slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_NEGATIVE), s);
  case ieee_floatt::ROUND_TO_PLUS_INF:
    return new_ast(slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_POSITIVE), s);
  case ieee_floatt::ROUND_TO_ZERO:
    return new_ast(slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_ZERO), s);
  default:
    break;
  }

  abort();
}

smt_astt cvc_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  std::size_t w = s->get_data_width();

  // Seems we can't make negative bitvectors; so just pull the value out and
  // assume CVC is going to cut the top off correctly.
  cvc5::Term e = slv.mkBitVector(w, theint.to_uint64());
  return new_ast(e, s);
}

smt_astt cvc_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = boolean_sort;
  cvc5::Term e = slv.mkBoolean(val);
  return new_ast(e, s);
}

smt_astt cvc_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt)
{
  return mk_smt_symbol(name, s);
}

smt_astt cvc_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  // Standard arrangement: if we already have the name, return the expression
  // from the symbol table. If not, time for a new name.

  /*if(sym_tab.isBound(name))
  {
    cvc5::Term e = sym_tab.lookup(name);
    return new_ast(e, s);
  }*/

  //AYB CVC5 no longer has a symbol table. Hopefully it does some silent checking internally
  // if not, we may need to implement a symbol table here

  // Time for a new one.
  cvc5::Term e =
    slv.mkConst(to_solver_smt_sort<cvc5::Sort>(s)->s, name); // "global", eh?
  //sym_tab.bind(name, e, true);
  return new_ast(e, s);
}

smt_astt cvc_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  // If it's a floatbv, convert it to bv
  if (a->sort->id == SMT_SORT_FPBV)
    a = mk_from_fp_to_bv(a);

  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);

  auto op = slv.mkOp(cvc5::Kind::BITVECTOR_EXTRACT, {high, low});
  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(high - low + 1);
  return new_ast(fin, s);
}

smt_astt cvc_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);

  auto op = slv.mkOp(cvc5::Kind::BITVECTOR_SIGN_EXTEND, {topwidth});

  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(fin, s);
}

smt_astt cvc_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  auto const *ca = to_solver_smt_ast<cvc_smt_ast>(a);

  auto op = slv.mkOp(cvc5::Kind::BITVECTOR_ZERO_EXTEND, {topwidth});
  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(fin, s);
}

smt_astt cvc_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  cvc5::Term e = slv.mkTerm(
    cvc5::Kind::BITVECTOR_CONCAT,
    {to_solver_smt_ast<cvc_smt_ast>(a)->a,
     to_solver_smt_ast<cvc_smt_ast>(b)->a});

  return new_ast(e, s);
}

smt_astt cvc_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  cvc5::Term e = slv.mkTerm(
    cvc5::Kind::ITE,
    {to_solver_smt_ast<cvc_smt_ast>(cond)->a,
     to_solver_smt_ast<cvc_smt_ast>(t)->a,
     to_solver_smt_ast<cvc_smt_ast>(f)->a});

  return new_ast(e, t->sort);
}

smt_astt
cvc_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

smt_sortt cvc_convt::mk_bool_sort()
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BOOL, slv.getBooleanSort(), 1);
}

smt_sortt cvc_convt::mk_real_sort()
{
  return new solver_smt_sort<cvc5::Sort>(SMT_SORT_REAL, slv.getRealSort());
}

smt_sortt cvc_convt::mk_int_sort()
{
  return new solver_smt_sort<cvc5::Sort>(SMT_SORT_INT, slv.getIntegerSort());
}

smt_sortt cvc_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BV, slv.mkBitVectorSort(width), width);
}

smt_sortt cvc_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FIXEDBV, slv.mkBitVectorSort(width), width);
}

smt_sortt cvc_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<cvc5::Sort>(domain);
  auto range_sort = to_solver_smt_sort<cvc5::Sort>(range);

  auto t = slv.mkArraySort(domain_sort->s, range_sort->s);
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_sortt cvc_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BVFP, slv.mkBitVectorSort(ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt cvc_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BVFP_RM, slv.mkBitVectorSort(3), 3);
}

smt_sortt cvc_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FPBV, slv.mkFloatingPointSort(ew, sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt cvc_convt::mk_fpbv_rm_sort()
{
  // TODO: Not sure about this
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FPBV_RM, slv.getRoundingModeSort(), 3);
}

void cvc_convt::dump_smt()
{
  std::ostringstream oss;
  auto const &assertions = slv.getAssertions();
  for (auto const &a : assertions)
    oss << a.toString();
  log_status("{}", oss.str());
}

void cvc_smt_ast::dump() const
{
  std::ostringstream oss;
  log_status("{}", a.toString());
}
