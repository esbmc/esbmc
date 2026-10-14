#include <cstdint>
#include <util/lang/c_types.h>
#include <util/arith/mp_arith.h>
#include <cvc5_conv.h>
#include <solvers/solve.h>

#define new_ast new_solver_ast<cvc5_smt_ast>

smt_solver_baset *create_new_cvc5_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  cvc5_convt *conv = new cvc5_convt(ns, options);
  // Native datatype-based tuple encoding is opt-in: defaulting to it
  // regresses 256-bit BV-heavy queries (e.g. typedef_1, modifier_named_*)
  // because CVC5's BV solver handles flattened pointer/struct tuples faster
  // than the datatype theory. Users hitting the bare-smt_sort abort on
  // Solidity nested-dynamic shapes (T[N][infinite]) opt in via
  // --cvc5-native-tuples; without the flag CVC5 falls back to the
  // smt_tuple_node_flattener for byte-identical pre-Stage-2 behavior.
  if (options.get_bool_option("cvc5-native-tuples"))
    *tuple_api = static_cast<tuple_iface *>(conv);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

cvc5_convt::cvc5_convt(const namespacet &ns, const optionst &options)
  : smt_solver_baset(ns, options),
    array_iface(true, true),
    fp_convt(this),
    to_bv_counter(0),
    slv()
{
  // Already initialized stuff in the constructor list,
  slv.setOption("produce-models", "true");
  slv.setOption("produce-assertions", "true");
  // Per-query budget (--path-cov-claim-timeout). `tlimit-per` is cvc5's
  // per-check-sat wall-clock limit in milliseconds, as opposed to `tlimit`,
  // which is cumulative over the solver's whole lifetime -- the wrong one here,
  // because a path-coverage run reuses one solver instance under
  // --smt-during-symex and a cumulative limit would abandon every claim after
  // the budget rather than each claim that exceeds it.
  if (const uint64_t ms = smt_per_query_timeout_ms(options))
  {
    slv.setOption("tlimit-per", std::to_string(ms));
    smt_record_timeout_mechanism(
      "cvc5: native option `tlimit-per` (per check-sat, milliseconds)");
  }
}

smt_resultt cvc5_convt::dec_solve()
{
  pre_solve();

  cvc5::Result r = slv.checkSat();
  if (r.isSat())
    return P_SATISFIABLE;

  if (r.isUnknown())
    return P_ERROR;

  return P_UNSATISFIABLE;
}

tvt cvc5_convt::get_bool(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);
  return tvt(e.getBooleanValue());
}

ieee_floatt cvc5_convt::get_fpbv(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);

  /* widths of exponent, significand and the bitvector value as a cvc5::Term */
  auto [ew, sw, bv] = e.getFloatingPointValue();

  assert(ew == a->sort->get_exponent_width());
  assert(sw == a->sort->get_significand_width());

  /* the sign bit isn't part of the significand's width for ieee_float_spect */
  ieee_floatt number(ieee_float_spect(sw - 1, ew));

  if (e.isFloatingPointNaN())
    number.make_NaN();
  else if (e.isFloatingPointPosInf())
    number.make_plus_infinity();
  else if (e.isFloatingPointNegInf())
    number.make_minus_infinity();
  else
  {
    std::string str = bv.getBitVectorValue(2);
    BigInt value = binary2integer(str, false);
    number.unpack(value);
  }

  return number;
}

BigInt cvc5_convt::get_bv(smt_astt a, bool is_signed)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);
  cvc5::Term e = slv.getValue(ca->a);
  std::string str = e.getBitVectorValue();
  return binary2integer(str, is_signed);
}

expr2tc cvc5_convt::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  auto const *carray = to_solver_smt_ast<cvc5_smt_ast>(array);
  size_t orig_w = array->sort->get_domain_width();

  smt_astt tmpast = mk_smt_bv(BigInt(index), mk_bv_sort(orig_w));
  auto const *tmpa = to_solver_smt_ast<cvc5_smt_ast>(tmpast);
  cvc5::Term e = slv.mkTerm(cvc5::Kind::SELECT, {carray->a, tmpa->a});
  delete tmpast;

  return get_by_ast(subtype, new_ast(e, convert_sort(subtype)));
}

const std::string cvc5_convt::solver_text()
{
  std::stringstream ss;
  ss << "CVC5 " << slv.getVersion();
  return ss.str();
}

smt_astt cvc5_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::ADD,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::SUB,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::MULT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_mod(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::INTS_MODULUS,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_div(smt_astt a, smt_astt b)
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
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_shl(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);

  cvc5::Term p =
    slv.mkTerm(cvc5::Kind::POW2, {to_solver_smt_ast<cvc5_smt_ast>(b)->a});
  return new_ast(
    slv.mkTerm(cvc5::Kind::MULT, {to_solver_smt_ast<cvc5_smt_ast>(a)->a, p}),
    a->sort);
}

smt_astt cvc5_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::NEG, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::LT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::GT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::LEQ,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::GEQ,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_real2int(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::TO_INTEGER, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_int2real(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT);
  return new_ast(
    slv.mkTerm(cvc5::Kind::TO_REAL, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_isint(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::IS_INTEGER, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt
cvc5_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_FMA,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(v1)->a,
       to_solver_smt_ast<cvc5_smt_ast>(v2)->a,
       to_solver_smt_ast<cvc5_smt_ast>(v3)->a}),
    v1->sort);
}

smt_astt
cvc5_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const cvc5_smt_ast *mrm =
    to_solver_smt_ast<cvc5_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const cvc5_smt_ast *mfrom = to_solver_smt_ast<cvc5_smt_ast>(from);

  /* The cvc5::Op API works with uint32_t indices for operators */
  assert(width <= UINT32_MAX);
  uint32_t w32 = width;
  cvc5::Op op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_UBV, {w32});

  return new_ast(slv.mkTerm(op, {mrm->a, mfrom->a}), mk_bv_sort(width));
}

smt_astt
cvc5_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const cvc5_smt_ast *mrm =
    to_solver_smt_ast<cvc5_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const cvc5_smt_ast *mfrom = to_solver_smt_ast<cvc5_smt_ast>(from);

  /* The cvc5::Op API works with uint32_t indices for operators */
  assert(width <= UINT32_MAX);
  uint32_t w32 = width;
  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_SBV, {w32});

  return new_ast(slv.mkTerm(op, {mrm->a, mfrom->a}), mk_bv_sort(width));
}

smt_astt cvc5_convt::mk_smt_typecast_from_fpbv_to_fpbv(
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
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(from)->a}),
    to);
}

smt_astt cvc5_convt::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_UBV, {ew, sw});

  return new_ast(
    slv.mkTerm(
      op,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(from)->a}),
    to);
}

smt_astt cvc5_convt::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_SBV, {ew, sw});

  return new_ast(
    slv.mkTerm(
      op,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(from)->a}),
    to);
}

smt_astt cvc5_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_exponent_width();

  auto op2 = slv.mkOp(cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {ew, sw});

  return new_ast(slv.mkTerm(op2, {to_solver_smt_ast<cvc5_smt_ast>(op)->a}), to);
}

smt_astt cvc5_convt::mk_from_fp_to_bv(smt_astt op)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(op);

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

smt_astt cvc5_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_ADD,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_SUB,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_MULT,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_DIV,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_rem(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_REM,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt cvc5_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_RTI,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(from)->a}),
    from->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_SQRT,
      {to_solver_smt_ast<cvc5_smt_ast>(rm)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rd)->a}),
    rd->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_EQ,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_GT,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_LT,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_GEQ,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_LEQ,
      {to_solver_smt_ast<cvc5_smt_ast>(lhs)->a,
       to_solver_smt_ast<cvc5_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NAN,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_INF,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NORMAL,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_ZERO,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_NEG,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_IS_POS,
      {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_ABS, {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    op->sort);
}

smt_astt cvc5_convt::mk_smt_fpbv_neg(smt_astt op)
{
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::FLOATINGPOINT_NEG, {to_solver_smt_ast<cvc5_smt_ast>(op)->a}),
    op->sort);
}

void cvc5_convt::assert_ast(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);
  slv.assertFormula(ca->a);
}

smt_astt cvc5_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ADD,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SUB,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_MULT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SREM,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UREM,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SDIV,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UDIV,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SHL,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ASHR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_LSHR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NEG, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_NOT, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_XOR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_OR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_AND,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::IMPLIES,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::XOR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::OR,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::AND,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    slv.mkTerm(cvc5::Kind::NOT, {to_solver_smt_ast<cvc5_smt_ast>(a)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ULT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SLT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UGT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SGT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_ULE,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SLE,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_UGE,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::BITVECTOR_SGE,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::EQUAL,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::DISTINCT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt cvc5_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::STORE,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a,
       to_solver_smt_ast<cvc5_smt_ast>(c)->a}),
    a->sort);
}

smt_astt cvc5_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    slv.mkTerm(
      cvc5::Kind::SELECT,
      {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
       to_solver_smt_ast<cvc5_smt_ast>(b)->a}),
    a->sort->get_range_sort());
}

smt_astt cvc5_convt::mk_smt_int(const BigInt &theint)
{
  // BigInt::to_int64 silently truncates past 64 bits, so for values outside
  // the int64 range fall back to cvc5::Solver::mkInteger(const std::string&)
  // which accepts an arbitrary-precision decimal literal. Issue #4642.
  cvc5::Term e = theint.is_int64() ? slv.mkInteger(theint.to_int64())
                                   : slv.mkInteger(integer2string(theint, 10));
  return new_ast(e, mk_int_sort());
}

smt_astt cvc5_convt::mk_smt_real(const std::string &str)
{
  smt_sortt s = mk_real_sort();
  return new_ast(slv.mkReal(str), s);
}

smt_astt cvc5_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  smt_sortt s = mk_real_fp_sort(thereal.spec.e, thereal.spec.f);
  BigInt fp_rep = thereal.pack();

  unsigned width = thereal.spec.width();
  cvc5::Term float_bv;
  if (thereal.spec.width() <= 64)
    float_bv = slv.mkBitVector(width, fp_rep.to_uint64());
  else
    float_bv = slv.mkBitVector(width, integer2binary(fp_rep, width), 2);

  return new_ast(
    slv.mkFloatingPoint(
      s->get_exponent_width(), s->get_significand_width(), float_bv),
    s);
}

smt_astt cvc5_convt::mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw - 1);
  smt_astt the_nan = new_ast(
    slv.mkFloatingPointNaN(s->get_exponent_width(), s->get_significand_width()),
    s);

  if (sgn)
    the_nan = fp_convt::mk_smt_fpbv_neg(the_nan);

  return the_nan;
}

smt_astt cvc5_convt::mk_smt_fpbv_inf(bool sign, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw - 1);

  cvc5::Term t = sign ? slv.mkFloatingPointNegInf(ew, sw)
                      : slv.mkFloatingPointPosInf(ew, sw);

  return new_ast(t, s);
}

smt_astt cvc5_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  smt_sortt s = mk_fpbv_rm_sort();

  switch (rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    return new_ast(
      slv.mkRoundingMode(cvc5::RoundingMode::ROUND_NEAREST_TIES_TO_EVEN), s);
  case ieee_floatt::ROUND_TO_MINUS_INF:
    return new_ast(
      slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_NEGATIVE), s);
  case ieee_floatt::ROUND_TO_PLUS_INF:
    return new_ast(
      slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_POSITIVE), s);
  case ieee_floatt::ROUND_TO_ZERO:
    return new_ast(
      slv.mkRoundingMode(cvc5::RoundingMode::ROUND_TOWARD_ZERO), s);
  default:
    break;
  }

  abort();
}

smt_astt cvc5_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  std::size_t w = s->get_data_width();

  cvc5::Term bv;
  if (theint.is_negative() || !theint.is_uint64())
    bv = slv.mkBitVector(w, integer2binary(theint, w), 2);
  else
    bv = slv.mkBitVector(w, theint.to_uint64());

  return new_ast(bv, s);
}

smt_astt cvc5_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = boolean_sort;
  cvc5::Term e = slv.mkBoolean(val);
  return new_ast(e, s);
}

smt_astt cvc5_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt)
{
  return mk_smt_symbol(name, s);
}

smt_astt cvc5_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  // Standard arrangement: if we already have the name, return the expression
  // from the symbol table. If not, time for a new name.

  symtabt::iterator it = symtable.find(name);
  if (it != symtable.end())
    return it->ast;

  // Time for a new one.
  const cvc5::Sort &srt = to_solver_smt_sort<cvc5::Sort>(s)->s;
  cvc5::Term e = slv.mkConst(srt, name); // "global", eh?
  smt_astt ast = new_ast(e, s);

  symtable.emplace(name, ast, ctx_level);

  return ast;
}

smt_astt cvc5_convt::mk_smt_uninterpreted_function(
  const std::string &name,
  const std::vector<smt_astt> &args,
  smt_sortt rangesort)
{
  // A nullary uninterpreted function is just a fixed constant; mk_smt_symbol
  // already caches it by name, so repeated uses share one term (congruence).
  if (args.empty())
    return mk_smt_symbol(name, rangesort);

  // Declare-or-reuse the function constant so every application shares it and
  // the solver enforces functional congruence natively.
  auto it = uf_decls.find(name);
  cvc5::Term fun;
  if (it != uf_decls.end())
    fun = it->second;
  else
  {
    std::vector<cvc5::Sort> domain;
    domain.reserve(args.size());
    for (smt_astt arg : args)
      domain.push_back(to_solver_smt_sort<cvc5::Sort>(arg->sort)->s);

    cvc5::Sort fun_sort =
      slv.mkFunctionSort(domain, to_solver_smt_sort<cvc5::Sort>(rangesort)->s);
    fun = slv.mkConst(fun_sort, name);
    uf_decls.emplace(name, fun);
  }

  // APPLY_UF takes [function, arg0, arg1, ...].
  std::vector<cvc5::Term> apply_args;
  apply_args.reserve(args.size() + 1);
  apply_args.push_back(fun);
  for (smt_astt arg : args)
    apply_args.push_back(to_solver_smt_ast<cvc5_smt_ast>(arg)->a);

  return new_ast(slv.mkTerm(cvc5::Kind::APPLY_UF, apply_args), rangesort);
}

smt_astt cvc5_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  // If it's a floatbv, convert it to bv
  if (a->sort->id == SMT_SORT_FPBV)
    a = mk_from_fp_to_bv(a);

  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);

  cvc5::Op op = slv.mkOp(cvc5::Kind::BITVECTOR_EXTRACT, {high, low});
  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(high - low + 1);
  return new_ast(fin, s);
}

smt_astt cvc5_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);

  cvc5::Op op = slv.mkOp(cvc5::Kind::BITVECTOR_SIGN_EXTEND, {topwidth});
  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(fin, s);
}

smt_astt cvc5_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  auto const *ca = to_solver_smt_ast<cvc5_smt_ast>(a);

  cvc5::Op op = slv.mkOp(cvc5::Kind::BITVECTOR_ZERO_EXTEND, {topwidth});
  cvc5::Term fin = slv.mkTerm(op, {ca->a});

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(fin, s);
}

smt_astt cvc5_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  cvc5::Term e = slv.mkTerm(
    cvc5::Kind::BITVECTOR_CONCAT,
    {to_solver_smt_ast<cvc5_smt_ast>(a)->a,
     to_solver_smt_ast<cvc5_smt_ast>(b)->a});

  return new_ast(e, s);
}

smt_astt cvc5_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  cvc5::Term e = slv.mkTerm(
    cvc5::Kind::ITE,
    {to_solver_smt_ast<cvc5_smt_ast>(cond)->a,
     to_solver_smt_ast<cvc5_smt_ast>(t)->a,
     to_solver_smt_ast<cvc5_smt_ast>(f)->a});

  return new_ast(e, t->sort);
}

smt_astt
cvc5_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  smt_sortt dom = mk_int_bv_sort(domain_width);
  smt_sortt arr = mk_array_sort(dom, init_val->sort);
  cvc5::Sort arr_sort = to_solver_smt_sort<cvc5::Sort>(arr)->s;
  cvc5::Term init_term = to_solver_smt_ast<cvc5_smt_ast>(init_val)->a;

  // CVC5's mkConstArray requires `val` to be a true SMT-LIB value (a literal),
  // unlike Z3's Z3_mk_const_array which happily accepts any well-sorted Term
  // including free constants and applied operators. When init_term is not a
  // value (e.g. a fresh tuple constant for a `NULL` pointer, or a chain of
  // `(store ...)` terms produced by tuple_array_create's finite fallback),
  // mkConstArray throws `Invalid argument 'NULL'/'<term>' for 'val', expected
  // a value`. Fall back to a fresh array constant in that case — the
  // resulting array has unconstrained slot values rather than all-slots-equal,
  // which is sound (more permissive) for default storage initialization where
  // the symex dispatcher loop havocs the storage on every iteration anyway.
  cvc5::Term t;
  try
  {
    t = slv.mkConstArray(arr_sort, init_term);
  }
  catch (const cvc5::CVC5ApiException &)
  {
    std::string name = mk_fresh_name("cvc5_convt::convert_array_of_fresh");
    t = slv.mkConst(arr_sort, name);
  }

  return new_ast(t, arr);
}

smt_sortt cvc5_convt::mk_bool_sort()
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BOOL, slv.getBooleanSort(), 1);
}

smt_sortt cvc5_convt::mk_real_sort()
{
  return new solver_smt_sort<cvc5::Sort>(SMT_SORT_REAL, slv.getRealSort());
}

smt_sortt cvc5_convt::mk_int_sort()
{
  return new solver_smt_sort<cvc5::Sort>(SMT_SORT_INT, slv.getIntegerSort());
}

smt_sortt cvc5_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BV, slv.mkBitVectorSort(width), width);
}

smt_sortt cvc5_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FIXEDBV, slv.mkBitVectorSort(width), width);
}

smt_sortt cvc5_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<cvc5::Sort>(domain);
  auto range_sort = to_solver_smt_sort<cvc5::Sort>(range);

  auto t = slv.mkArraySort(domain_sort->s, range_sort->s);
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_sortt cvc5_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BVFP, slv.mkBitVectorSort(ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt cvc5_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_BVFP_RM, slv.mkBitVectorSort(3), 3);
}

smt_sortt cvc5_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FPBV, slv.mkFloatingPointSort(ew, sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt cvc5_convt::mk_fpbv_rm_sort()
{
  // TODO: Not sure about this
  return new solver_smt_sort<cvc5::Sort>(
    SMT_SORT_FPBV_RM, slv.getRoundingModeSort(), 3);
}

void cvc5_convt::push_ctx()
{
  smt_solver_baset::push_ctx();
  slv.push();
}

void cvc5_convt::pop_ctx()
{
  symtabt::nth_index<1>::type &symtab_levels = symtable.get<1>();
  symtab_levels.erase(ctx_level);

  slv.pop();
  smt_solver_baset::pop_ctx();
}

std::string cvc5_convt::dump_smt()
{
  std::ostringstream oss;
  auto const &assertions = slv.getAssertions();
  for (auto const &a : assertions)
    oss << a.toString();
  return oss.str();
}

void cvc5_smt_ast::dump() const
{
  std::ostringstream oss;
  log_status("{}", a.toString());
}

smt_astt cvc5_convt::mk_quantifier(
  bool is_forall,
  std::vector<smt_astt> lhs,
  smt_astt rhs)
{
  if (lhs.empty())
  {
    return rhs;
  }
  using namespace cvc5;

  quantifier_counter++;

  std::vector<Term> bound_vars;
  std::vector<Term> original_vars;

  for (size_t i = 0; i < lhs.size(); i++)
  {
    Term orig = to_solver_smt_ast<cvc5_smt_ast>(lhs[i])->a;
    original_vars.push_back(orig);
    Sort sort = orig.getSort();
    std::string name =
      "qvar_" + std::to_string(quantifier_counter) + "_" + std::to_string(i);
    Term bound = slv.mkVar(sort, name);
    bound_vars.push_back(bound);
  }

  // Here the body term is created.
  Term body = to_solver_smt_ast<cvc5_smt_ast>(rhs)->a;

  // Here the substitution is performed.
  Term substituted_body = body.substitute(original_vars, bound_vars);

  // Here the respective quantifier is created.
  Term vars_list = slv.mkTerm(cvc5::Kind::VARIABLE_LIST, bound_vars);

  Term quantifier;
  if (is_forall)
  {
    quantifier = slv.mkTerm(cvc5::Kind::FORALL, {vars_list, substituted_body});
  }
  else
  {
    quantifier = slv.mkTerm(cvc5::Kind::EXISTS, {vars_list, substituted_body});
  }

  return new_ast(quantifier, rhs->sort);
}

// ---------------------------------------------------------------------------
// Native tuple support via CVC5 datatype/tuple API.  Mirrors z3_conv.cpp's
// implementation so cvc5_convt acts as its own tuple_iface (avoiding the
// smt_tuple_node_flattener fallback that produces bare smt_sort instances).
// ---------------------------------------------------------------------------

cvc5::Term cvc5_convt::cvc5_tuple_select(const cvc5::Term &t, unsigned i)
{
  cvc5::Sort srt = t.getSort();
  if (!srt.isTuple())
  {
    log_error("cvc5 conversion: argument must be a tuple");
    abort();
  }
  cvc5::Datatype dt = srt.getDatatype();
  // Tuples are single-constructor datatypes; field selectors live on dt[0].
  cvc5::DatatypeSelector sel = dt[0][i];
  return slv.mkTerm(cvc5::Kind::APPLY_SELECTOR, {sel.getTerm(), t});
}

cvc5::Term cvc5_convt::cvc5_tuple_update(
  const cvc5::Term &t,
  unsigned i,
  const cvc5::Term &newval)
{
  cvc5::Sort srt = t.getSort();
  if (!srt.isTuple())
  {
    log_error("cvc5 conversion: argument must be a tuple");
    abort();
  }
  cvc5::Datatype dt = srt.getDatatype();
  cvc5::DatatypeSelector sel = dt[0][i];
  return slv.mkTerm(
    cvc5::Kind::APPLY_UPDATER, {sel.getUpdaterTerm(), t, newval});
}

smt_astt cvc5_smt_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  if (sort->id == SMT_SORT_ARRAY)
    return smt_ast::update(ctx, value, idx, idx_expr);

  assert(sort->id == SMT_SORT_STRUCT);
  assert(is_nil_expr(idx_expr) && "Can only update constant index tuple elems");

  cvc5_convt *cv = static_cast<cvc5_convt *>(ctx);
  const cvc5_smt_ast *updateval = to_solver_smt_ast<cvc5_smt_ast>(value);
  return cv->new_solver_ast<cvc5_smt_ast>(
    cv->cvc5_tuple_update(a, idx, updateval->a), sort);
}

smt_astt
cvc5_smt_ast::project(smt_solver_baset *ctx, unsigned int elem) const
{
  cvc5_convt *cv = static_cast<cvc5_convt *>(ctx);

  assert(!is_nil_type(sort->get_tuple_type()));
  const std::vector<type2tc> &members =
    struct_union_members(sort->get_tuple_type());

  assert(elem < members.size());
  smt_sortt idx_sort = ctx->convert_sort(members[elem]);

  cvc5::Term selected = cv->cvc5_tuple_select(a, elem);
  return cv->new_solver_ast<cvc5_smt_ast>(selected, idx_sort);
}

smt_sortt cvc5_convt::mk_struct_sort(const type2tc &type)
{
  // Match z3_convt::mk_struct_sort: array-of-tuple gets emitted as a regular
  // array sort, and the carve-out at smt_conv.cpp's array_id case routes us
  // here on the tuple-typed range path.
  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    smt_sortt subtypesort = convert_sort(arrtype.subtype);
    smt_sortt d = mk_int_bv_sort(make_array_domain_type(arrtype)->get_width());
    return mk_array_sort(d, subtypesort);
  }

  const std::vector<type2tc> &def_members =
    struct_union_members(is_pointer_type(type) ? pointer_struct : type);
  std::vector<cvc5::Sort> sorts;
  sorts.reserve(def_members.size());
  for (auto const &it : def_members)
  {
    smt_sortt s = convert_sort(it);
    sorts.push_back(to_solver_smt_sort<cvc5::Sort>(s)->s);
  }

  cvc5::Sort tuple_sort = slv.mkTupleSort(sorts);
  return new solver_smt_sort<cvc5::Sort>(SMT_SORT_STRUCT, tuple_sort, type);
}

smt_astt cvc5_convt::tuple_create(const expr2tc &structdef)
{
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const std::vector<expr2tc> &members = strct.datatype_members;
  const std::size_t n_members = struct_union_members(strct.type).size();

  std::vector<cvc5::Term> args;
  args.reserve(n_members);
  for (std::size_t i = 0; i < n_members; ++i)
    args.push_back(to_solver_smt_ast<cvc5_smt_ast>(convert_ast(members[i]))->a);

  smt_sortt s = mk_struct_sort(structdef->type);
  cvc5::Term t = slv.mkTuple(args);
  return new_ast(t, s);
}

smt_astt cvc5_convt::tuple_fresh(smt_sortt s, std::string name)
{
  if (name.empty())
    name = mk_fresh_name("cvc5_convt::tuple_fresh");
  cvc5::Term t = slv.mkConst(to_solver_smt_sort<cvc5::Sort>(s)->s, name);
  return new_ast(t, s);
}

smt_astt cvc5_convt::tuple_array_create(
  const type2tc &arr_type,
  smt_astt *input_args,
  bool const_array,
  smt_sortt domain)
{
  const array_type2t &arrtype = to_array_type(arr_type);

  smt_sortt ssort = mk_struct_sort(arrtype.subtype);
  smt_sortt asort = mk_array_sort(domain, ssort);
  cvc5::Sort cvc5_arr_sort = to_solver_smt_sort<cvc5::Sort>(asort)->s;
  std::size_t dom_w = domain->get_data_width();

  if (const_array)
  {
    const cvc5_smt_ast *tmpast = to_solver_smt_ast<cvc5_smt_ast>(*input_args);
    cvc5::Term value = tmpast->a;
    if (is_bool_type(arrtype.subtype))
      value = slv.mkBoolean(false);

    // Z3 happily accepts non-value Terms (free constants, applied selectors)
    // as the default for Z3_mk_const_array. CVC5's mkConstArray is stricter —
    // it requires the default to actually be a value. For finite tuple arrays
    // (the common case for Solidity nested-dynamic shapes after rewrite_ptrs_
    // to_structs), iterate per-slot stores from a fresh array constant; this
    // pins every reachable index to `value` without the value-vs-constant
    // distinction.
    if (
      !arrtype.size_is_infinite && !is_nil_expr(arrtype.array_size) &&
      is_constant_int2t(arrtype.array_size))
    {
      std::string name =
        mk_fresh_name("cvc5_convt::tuple_array_create_const");
      cvc5::Term out = slv.mkConst(cvc5_arr_sort, name);
      std::size_t N = to_constant_int2t(arrtype.array_size).as_ulong();
      for (std::size_t i = 0; i < N; ++i)
      {
        cvc5::Term idx = slv.mkBitVector(dom_w, i);
        out = slv.mkTerm(cvc5::Kind::STORE, {out, idx, value});
      }
      return new_ast(out, asort);
    }

    // Infinite-size const_array path: must use mkConstArray (no per-slot
    // iteration possible). Same value-vs-free-constant strictness applies;
    // fall back to a fresh array constant when CVC5 rejects `value` as a
    // non-value Term. Sound (more permissive) for default storage init.
    cvc5::Term t;
    try
    {
      t = slv.mkConstArray(cvc5_arr_sort, value);
    }
    catch (const cvc5::CVC5ApiException &)
    {
      std::string name =
        mk_fresh_name("cvc5_convt::tuple_array_create_inf_fresh");
      t = slv.mkConst(cvc5_arr_sort, name);
    }
    return new_ast(t, asort);
  }

  assert(
    !is_nil_expr(arrtype.array_size) &&
    "Non-const tuple_array_create with infinite array size");
  assert(
    is_constant_int2t(arrtype.array_size) &&
    "tuple_array_create sizes should be constant");

  std::string name = mk_fresh_name("cvc5_convt::tuple_array_create");
  cvc5::Term out = slv.mkConst(cvc5_arr_sort, name);
  for (std::size_t i = 0; i < to_constant_int2t(arrtype.array_size).as_ulong();
       ++i)
  {
    cvc5::Term idx = slv.mkBitVector(dom_w, i);
    const cvc5_smt_ast *tmpast = to_solver_smt_ast<cvc5_smt_ast>(input_args[i]);
    out = slv.mkTerm(cvc5::Kind::STORE, {out, idx, tmpast->a});
  }

  return new_ast(out, asort);
}

smt_astt cvc5_convt::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_astt cvc5_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
}

smt_astt
cvc5_convt::tuple_array_of(const expr2tc &init, unsigned long domain_width)
{
  return convert_array_of(convert_ast(init), domain_width);
}

expr2tc cvc5_convt::tuple_get(const type2tc &type, smt_astt sym)
{
  const std::vector<type2tc> &members =
    struct_union_members(is_pointer_type(type) ? pointer_struct : type);

  if (is_pointer_type(type))
  {
    smt_astt object = new_ast(
      cvc5_tuple_select(to_solver_smt_ast<cvc5_smt_ast>(sym)->a, 0),
      convert_sort(members[0]));
    smt_astt offset = new_ast(
      cvc5_tuple_select(to_solver_smt_ast<cvc5_smt_ast>(sym)->a, 1),
      convert_sort(members[1]));

    unsigned int num =
      get_bv(object, is_signedbv_type(members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, type);
  }

  std::vector<expr2tc> outmem;
  unsigned int i = 0;
  for (auto const &it : members)
  {
    outmem.push_back(get_by_ast(
      it,
      new_ast(
        cvc5_tuple_select(to_solver_smt_ast<cvc5_smt_ast>(sym)->a, i),
        convert_sort(it))));
    i++;
  }

  return constant_struct2tc(type, std::move(outmem));
}

expr2tc cvc5_convt::tuple_get(const expr2tc &expr)
{
  const type2tc &t = is_pointer_type(expr->type) ? pointer_struct : expr->type;
  const std::vector<type2tc> &members = struct_union_members(t);
  const std::vector<irep_idt> &mnames = struct_union_member_names(t);

  if (is_pointer_type(expr->type))
  {
    smt_astt sym = convert_ast(expr);

    smt_astt object = new_ast(
      cvc5_tuple_select(to_solver_smt_ast<cvc5_smt_ast>(sym)->a, 0),
      convert_sort(members[0]));
    smt_astt offset = new_ast(
      cvc5_tuple_select(to_solver_smt_ast<cvc5_smt_ast>(sym)->a, 1),
      convert_sort(members[1]));

    unsigned int num =
      get_bv(object, is_signedbv_type(members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  std::vector<expr2tc> outmem;
  unsigned int i = 0;
  for (auto const &it : members)
  {
    expr2tc memb = member2tc(it, expr, mnames[i]);
    outmem.push_back(get(memb));
    i++;
  }

  return constant_struct2tc(expr->type, std::move(outmem));
}

expr2tc cvc5_convt::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  return get_array_elem(array, index, get_flattened_array_subtype(subtype));
}
