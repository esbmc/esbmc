#include <bitwuzla_conv.h>
#include <cstdio>
#include <sstream>

#define new_ast new_solver_ast<bitw_smt_ast>

/* An fp sort and a bv sort are different Bitwuzla sorts, so the in-place
 * rewrite the base class performs would corrupt every other holder of this
 * ast. Hand back a fresh node over the same term instead. */
smt_astt bitw_smt_ast::with_sort(smt_solver_baset *ctx, smt_sortt s) const
{
  return ctx->new_solver_ast<bitw_smt_ast>(a, s);
}

namespace
{
/** The C API routed every Bitwuzla error through an abort callback
 *  (BITWUZLA_C_TRY_CATCH_END); the C++ API throws instead. Keep that failure
 *  mode at the sites a source program can reach, so a solver error still dies
 *  loudly rather than escaping as an exception some caller may read as an
 *  answer. */
template <typename callt>
auto guarded(callt call) -> decltype(call())
{
  try
  {
    return call();
  }
  catch (const bitwuzla::Exception &e)
  {
    log_error("Bitwuzla error encountered\n{}", e.msg());
    abort();
  }
}
} // namespace

smt_solver_baset *create_new_bitwuzla_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api,
  fp_convt **fp_api)
{
  bitwuzla_convt *conv = new bitwuzla_convt(ns, options);
  *array_api = static_cast<array_iface *>(conv);
  /* --fp2bv opts back out to ESBMC's own bit-vector encoding, which is what
   * fp.rem-heavy programs and the sign of a NaN (#7021) still need. */
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

bitwuzla_convt::bitwuzla_convt(const namespacet &ns, const optionst &options)
  : smt_solver_baset(ns, options), array_iface(true, true), fp_convt(this)
{
  if (options.get_bool_option("int-encoding"))
  {
    log_error("Bitwuzla does not support integer encoding mode");
    abort();
  }

  bitw_options.set(bitwuzla::Option::PRODUCE_MODELS, 1);
  bitw = std::make_unique<bitwuzla::Bitwuzla>(tm, bitw_options);
}

bitwuzla_convt::~bitwuzla_convt() = default;

void bitwuzla_convt::push_ctx()
{
  smt_solver_baset::push_ctx();
  bitw->push(1);
}

void bitwuzla_convt::pop_ctx()
{
  symtabt::nth_index<1>::type &symtab_levels = symtable.get<1>();
  symtab_levels.erase(ctx_level);

  bitw->pop(1);
  smt_solver_baset::pop_ctx();
}

smt_resultt bitwuzla_convt::dec_solve()
{
  pre_solve();

  bitwuzla::Result result = bitw->check_sat();

  if (result == bitwuzla::Result::SAT)
    return P_SATISFIABLE;

  if (result == bitwuzla::Result::UNSAT)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string bitwuzla_convt::solver_text()
{
  std::string ss = "Bitwuzla ";
  ss += bitwuzla::version();
  return ss;
}

void bitwuzla_convt::assert_ast(smt_astt a)
{
  bitw->assert_formula(to_solver_smt_ast<bitw_smt_ast>(a)->a);
}

smt_astt bitwuzla_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_ADD,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SUB,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_MUL,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SREM,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_UREM,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SDIV,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_UDIV,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SHL,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_ASHR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SHR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    tm.mk_term(bitwuzla::Kind::BV_NEG, {to_solver_smt_ast<bitw_smt_ast>(a)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    tm.mk_term(bitwuzla::Kind::BV_NOT, {to_solver_smt_ast<bitw_smt_ast>(a)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_XOR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_OR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_AND,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::IMPLIES,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::XOR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::OR,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::AND,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    tm.mk_term(bitwuzla::Kind::NOT, {to_solver_smt_ast<bitw_smt_ast>(a)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_ULT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SLT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_UGT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SGT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_ULE,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SLE,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_UGE,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_SGE,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::EQUAL,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::DISTINCT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::ARRAY_STORE,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a,
       to_solver_smt_ast<bitw_smt_ast>(c)->a}),
    a->sort);
}

smt_astt bitwuzla_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::ARRAY_SELECT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    a->sort->get_range_sort());
}

smt_astt bitwuzla_convt::mk_smt_int(const BigInt &theint [[maybe_unused]])
{
  log_error("ESBMC can't create integer sorts with Bitwuzla yet");
  abort();
}

smt_astt bitwuzla_convt::mk_smt_real(const std::string &str [[maybe_unused]])
{
  log_error("ESBMC can't create real sorts with Bitwuzla yet");
  abort();
}

smt_astt bitwuzla_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  return new_ast(
    tm.mk_bv_value(
      to_solver_smt_sort<bitwuzla::Sort>(s)->s,
      integer2binary(theint, s->get_data_width()).c_str(),
      2),
    s);
}

smt_astt bitwuzla_convt::mk_smt_bool(bool val)
{
  bitwuzla::Term node = (val) ? tm.mk_true() : tm.mk_false();
  const smt_sort *sort = boolean_sort;
  return new_ast(node, sort);
}

smt_astt bitwuzla_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype [[maybe_unused]])
{
  return mk_smt_symbol(name, s);
}

smt_astt
bitwuzla_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  symtabt::iterator it = symtable.find(name);
  if (it != symtable.end())
    return it->ast;

  bitwuzla::Term node;

  switch (s->id)
  {
  case SMT_SORT_BV:
  case SMT_SORT_FIXEDBV:
  case SMT_SORT_BVFP:
  case SMT_SORT_BVFP_RM:
  case SMT_SORT_FPBV:
  case SMT_SORT_FPBV_RM:
  case SMT_SORT_BOOL:
  case SMT_SORT_ARRAY:
    node = tm.mk_const(to_solver_smt_sort<bitwuzla::Sort>(s)->s, name.c_str());
    break;

  default:
    log_error("Unknown type for symbol");
    abort();
  }

  smt_astt ast = new_ast(node, s);

  symtable.emplace(name, ast, ctx_level);

  return ast;
}

smt_astt bitwuzla_convt::mk_smt_uninterpreted_function(
  const std::string &name,
  const std::vector<smt_astt> &args,
  smt_sortt rangesort)
{
  // A nullary uninterpreted function is just a fixed constant; mk_smt_symbol
  // already caches it by name, so repeated uses share one term (congruence).
  if (args.empty())
    return mk_smt_symbol(name, rangesort);

  // Declare-or-reuse the function constant. Bitwuzla mints a fresh const on
  // every mk_const, so it is cached and reused across applications;
  // sharing one declaration is what makes the solver enforce congruence.
  auto it = uf_decls.find(name);
  bitwuzla::Term fun;
  if (it != uf_decls.end())
    fun = it->second;
  else
  {
    std::vector<bitwuzla::Sort> domain;
    domain.reserve(args.size());
    for (smt_astt arg : args)
      domain.push_back(to_solver_smt_sort<bitwuzla::Sort>(arg->sort)->s);

    fun = tm.mk_const(
      tm.mk_fun_sort(domain, to_solver_smt_sort<bitwuzla::Sort>(rangesort)->s),
      name);
    uf_decls.emplace(name, fun);
  }

  // bitwuzla::Kind::APPLY expects [function, arg0, arg1, ...].
  std::vector<bitwuzla::Term> apply_args;
  apply_args.reserve(args.size() + 1);
  apply_args.push_back(fun);
  for (smt_astt arg : args)
    apply_args.push_back(to_solver_smt_ast<bitw_smt_ast>(arg)->a);

  return new_ast(tm.mk_term(bitwuzla::Kind::APPLY, apply_args), rangesort);
}

smt_astt
bitwuzla_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  bitwuzla::Term b =
    tm.mk_term(bitwuzla::Kind::BV_EXTRACT, {ast->a}, {high, low});
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  bitwuzla::Term b =
    tm.mk_term(bitwuzla::Kind::BV_SIGN_EXTEND, {ast->a}, {topwidth});
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  bitwuzla::Term b =
    tm.mk_term(bitwuzla::Kind::BV_ZERO_EXTEND, {ast->a}, {topwidth});
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::BV_CONCAT,
      {to_solver_smt_ast<bitw_smt_ast>(a)->a,
       to_solver_smt_ast<bitw_smt_ast>(b)->a}),
    s);
}

smt_astt bitwuzla_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  // A float reaches here in either representation now: a native fp term where
  // the FP API produced it, a bit-vector where the bit-level paths did -- a
  // failed-dereference symbol merged against a byte-wise read, say. The widths
  // agree, and the bit-vector holds that format's IEEE encoding, so reinterpret
  // it rather than hand Bitwuzla an ite over two sorts, which it rejects.
  if (t->sort->id == SMT_SORT_FPBV && f->sort->id != SMT_SORT_FPBV)
    f = mk_from_bv_to_fp(f, t->sort);
  else if (f->sort->id == SMT_SORT_FPBV && t->sort->id != SMT_SORT_FPBV)
    t = mk_from_bv_to_fp(t, f->sort);

  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::ITE,
      {to_solver_smt_ast<bitw_smt_ast>(cond)->a,
       to_solver_smt_ast<bitw_smt_ast>(t)->a,
       to_solver_smt_ast<bitw_smt_ast>(f)->a}),
    t->sort);
}

tvt bitwuzla_convt::get_bool(smt_astt a)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  bitwuzla::Term value = bitw->get_value(ast->a);

  if (value.is_true())
    return tvt(true);
  if (value.is_false())
    return tvt(false);

  // Bitwuzla returns the query term unchanged when evaluating it would need a
  // quantifier it never registered, so there is no ground value here (#7063).
  log_debug(
    "solver", "Bitwuzla returned no boolean value; term is unevaluatable");
  return tvt(tvt::TV_UNKNOWN);
}

BigInt bitwuzla_convt::get_bv(smt_astt a, bool is_signed)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  return guarded([&] {
    return binary2integer(
      bitw->get_value(ast->a).value<std::string>(), is_signed);
  });
}

expr2tc bitwuzla_convt::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  const bitw_smt_ast *za = to_solver_smt_ast<bitw_smt_ast>(array);
  size_t array_bound = array->sort->get_domain_width();
  const bitw_smt_ast *idx;
  if (int_encoding)
    idx = to_solver_smt_ast<bitw_smt_ast>(mk_smt_int(BigInt(index)));
  else
    idx = to_solver_smt_ast<bitw_smt_ast>(
      mk_smt_bv(BigInt(index), mk_bv_sort(array_bound)));

  bitwuzla::Term array_value = bitw->get_value(za->a);
  bitwuzla::Term index_value = bitw->get_value(idx->a);

  bitwuzla::Term e =
    tm.mk_term(bitwuzla::Kind::ARRAY_SELECT, {array_value, index_value});

  return get_by_ast(subtype, new_ast(e, convert_sort(subtype)));
}

smt_astt bitwuzla_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  const expr2tc &op1 = *overflow.operand->get_sub_expr(0);
  const expr2tc &op2 = *overflow.operand->get_sub_expr(1);

  const bitw_smt_ast *side1 = to_solver_smt_ast<bitw_smt_ast>(convert_ast(op1));
  const bitw_smt_ast *side2 = to_solver_smt_ast<bitw_smt_ast>(convert_ast(op2));

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed = (is_signedbv_type(op1) || is_signedbv_type(op2));

  bitwuzla::Term res;
  if (is_add2t(overflow.operand))
  {
    if (is_signed)
    {
      res = tm.mk_term(bitwuzla::Kind::BV_SADD_OVERFLOW, {side1->a, side2->a});
    }
    else
    {
      res = tm.mk_term(bitwuzla::Kind::BV_UADD_OVERFLOW, {side1->a, side2->a});
    }
  }
  else if (is_sub2t(overflow.operand))
  {
    if (is_signed)
    {
      res = tm.mk_term(bitwuzla::Kind::BV_SSUB_OVERFLOW, {side1->a, side2->a});
    }
    else
    {
      res = tm.mk_term(bitwuzla::Kind::BV_USUB_OVERFLOW, {side1->a, side2->a});
    }
  }
  else if (is_mul2t(overflow.operand))
  {
    if (is_signed)
    {
      res = tm.mk_term(bitwuzla::Kind::BV_SMUL_OVERFLOW, {side1->a, side2->a});
    }
    else
    {
      res = tm.mk_term(bitwuzla::Kind::BV_UMUL_OVERFLOW, {side1->a, side2->a});
    }
  }
  else if (is_div2t(overflow.operand) || is_modulus2t(overflow.operand))
  {
    res = tm.mk_term(bitwuzla::Kind::BV_SDIV_OVERFLOW, {side1->a, side2->a});
  }
  else
  {
    return smt_solver_baset::overflow_arith(expr);
  }

  const smt_sort *s = boolean_sort;
  return new_ast(res, s);
}

smt_astt
bitwuzla_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  smt_sortt dom_sort = mk_int_bv_sort(domain_width);
  smt_sortt arrsort = mk_array_sort(dom_sort, init_val->sort);

  return new_ast(
    tm.mk_const_array(
      to_solver_smt_sort<bitwuzla::Sort>(arrsort)->s,
      to_solver_smt_ast<bitw_smt_ast>(init_val)->a),
    arrsort);
}

std::string bitwuzla_convt::dump_smt()
{
  std::ostringstream out;
  bitw->print_formula(out, "smt2");
  return out.str();
}

void bitw_smt_ast::dump() const
{
  log_status("{}", a.str());
}

void bitwuzla_convt::print_model()
{
  log_warning(
    "Bitwuzla model printing is experimental and does not guarantee correct "
    "results.");
  // TODO: We use `symbtable` to get all symbols, because there does not seem to
  // be a way to get all symbols from Bitwuzla. This is not ideal, because
  // I have no idea whether ignoring the `level` field, which is also part of
  // an entry in `symbtable`, is correct. However, it seems to work for now.
  for (const auto &entry : symtable)
  {
    const bitwuzla::Term &term = to_solver_smt_ast<bitw_smt_ast>(entry.ast)->a;
    bitwuzla::Sort sort = term.sort();
    auto symbol = term.symbol();
    fprintf(
      messaget::state.out,
      "(define-fun %s (",
      symbol ? symbol->get().c_str() : "");
    if (sort.is_fun())
    {
      bitwuzla::Term value = bitw->get_value(term);
      std::vector<bitwuzla::Term> children = value.children();
      assert(children.size() == 2);
      while (children[1].kind() == bitwuzla::Kind::LAMBDA)
      {
        assert(children[0].is_variable());
        fprintf(
          messaget::state.out,
          "(%s %s) ",
          children[0].str().c_str(),
          children[0].sort().str().c_str());
        value = children[1];
        children = value.children();
      }
      assert(children[0].is_variable());
      fprintf(
        messaget::state.out,
        "(%s %s)) %s %s)\n",
        children[0].str().c_str(),
        children[0].sort().str().c_str(),
        sort.fun_codomain().str().c_str(),
        children[1].str().c_str());
    }
    else
    {
      fprintf(
        messaget::state.out,
        ") %s %s)\n",
        sort.str().c_str(),
        bitw->get_value(term).str().c_str());
    }
  }
}

smt_sortt bitwuzla_convt::mk_bool_sort()
{
  return cached_sort({SMT_SORT_BOOL, 0, 0}, [this] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_BOOL, tm.mk_bool_sort(), 1);
  });
}

smt_sortt bitwuzla_convt::mk_bv_sort(std::size_t width)
{
  return cached_sort({SMT_SORT_BV, width, 0}, [this, width] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_BV, tm.mk_bv_sort(width), width);
  });
}

smt_sortt bitwuzla_convt::mk_fbv_sort(std::size_t width)
{
  return cached_sort({SMT_SORT_FIXEDBV, width, 0}, [this, width] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_FIXEDBV, tm.mk_bv_sort(width), width);
  });
}

smt_sortt bitwuzla_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  /* Keyed on the operand sorts' addresses: no smt_sort is ever freed (nothing
   * under src/solvers/ deletes one, and pop_ctx frees only smt_asts), so an
   * address can never be recycled into a different sort. Identity rather than
   * width because a Bool domain and a 1-bit bit-vector domain are distinct
   * Bitwuzla sorts that share a width. */
  sort_keyt key{
    SMT_SORT_ARRAY,
    reinterpret_cast<uintptr_t>(domain),
    reinterpret_cast<uintptr_t>(range)};

  return cached_sort(key, [this, domain, range] {
    auto domain_sort = to_solver_smt_sort<bitwuzla::Sort>(domain);
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_ARRAY,
      tm.mk_array_sort(
        domain_sort->s, to_solver_smt_sort<bitwuzla::Sort>(range)->s),
      domain_sort->get_data_width(),
      range);
  });
}

smt_sortt bitwuzla_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return cached_sort({SMT_SORT_BVFP, ew, sw}, [this, ew, sw] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_BVFP, tm.mk_bv_sort(ew + sw + 1), ew + sw + 1, sw + 1);
  });
}

smt_sortt bitwuzla_convt::mk_bvfp_rm_sort()
{
  return cached_sort({SMT_SORT_BVFP_RM, 0, 0}, [this] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_BVFP_RM, tm.mk_bv_sort(3), 3);
  });
}

smt_sortt bitwuzla_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  // sw excludes the hidden bit, which Bitwuzla's significand width includes.
  /* A source program can name a format Bitwuzla was not built with -- x87
   * long double under --32, say. */
  return cached_sort({SMT_SORT_FPBV, ew, sw}, [this, ew, sw] {
    return guarded([&] {
      return new solver_smt_sort<bitwuzla::Sort>(
        SMT_SORT_FPBV, tm.mk_fp_sort(ew, sw + 1), ew + sw + 1, sw + 1);
    });
  });
}

smt_sortt bitwuzla_convt::mk_fpbv_rm_sort()
{
  return cached_sort({SMT_SORT_FPBV_RM, 0, 0}, [this] {
    return new solver_smt_sort<bitwuzla::Sort>(
      SMT_SORT_FPBV_RM, tm.mk_rm_sort(), 3);
  });
}

smt_astt bitwuzla_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  smt_sortt s = mk_fpbv_sort(thereal.spec.e, thereal.spec.f);
  smt_astt bv =
    mk_smt_bv(thereal.pack(), mk_bvfp_sort(thereal.spec.e, thereal.spec.f));
  return mk_from_bv_to_fp(bv, s);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw)
{
  // SMT-LIB has a single NaN with no sign bit, so sgn cannot be honoured
  // here; observing the sign of a NaN is esbmc/esbmc#7021.
  (void)sgn;
  smt_sortt s = mk_fpbv_sort(ew, sw - 1);
  return new_ast(tm.mk_fp_nan(to_solver_smt_sort<bitwuzla::Sort>(s)->s), s);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_fpbv_sort(ew, sw - 1);
  bitwuzla::Sort bs = to_solver_smt_sort<bitwuzla::Sort>(s)->s;
  return new_ast(sgn ? tm.mk_fp_neg_inf(bs) : tm.mk_fp_pos_inf(bs), s);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  bitwuzla::RoundingMode brm;
  switch (rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    brm = bitwuzla::RoundingMode::RNE;
    break;
  case ieee_floatt::ROUND_TO_AWAY:
    brm = bitwuzla::RoundingMode::RNA;
    break;
  case ieee_floatt::ROUND_TO_PLUS_INF:
    brm = bitwuzla::RoundingMode::RTP;
    break;
  case ieee_floatt::ROUND_TO_MINUS_INF:
    brm = bitwuzla::RoundingMode::RTN;
    break;
  case ieee_floatt::ROUND_TO_ZERO:
    brm = bitwuzla::RoundingMode::RTZ;
    break;
  default:
    log_error("Unexpected rounding mode reached Bitwuzla");
    abort();
  }

  return new_ast(tm.mk_rm_value(brm), mk_fpbv_rm_sort());
}

smt_astt bitwuzla_convt::mk_fp_arith(
  bitwuzla::Kind kind,
  smt_astt lhs,
  smt_astt rhs,
  smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      kind,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(lhs)->a,
       to_solver_smt_ast<bitw_smt_ast>(rhs)->a}),
    lhs->sort);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(bitwuzla::Kind::FP_ADD, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(bitwuzla::Kind::FP_SUB, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(bitwuzla::Kind::FP_MUL, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(bitwuzla::Kind::FP_DIV, lhs, rhs, rm);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_rem(smt_astt lhs, smt_astt rhs)
{
  /* Bitwuzla has fp.rem, but it solves the remainder/fmod bound proofs one to
   * two orders of magnitude slower than ESBMC's own lowering, so round-trip to
   * bit-vectors and use that instead, as the mathsat backend does. A separate
   * fp_convt is needed rather than fp_convt::mk_smt_fpbv_rem: the lowering
   * calls back into the interface, and through *this* those calls would reach
   * the native overrides and hand FP terms to bit-vector operations. */
  fp_convt software(this);
  smt_astt rem =
    software.mk_smt_fpbv_rem(mk_from_fp_to_bv(lhs), mk_from_fp_to_bv(rhs));
  return mk_from_bv_to_fp(rem, lhs->sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_fma(
  smt_astt v1,
  smt_astt v2,
  smt_astt v3,
  smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_FMA,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(v1)->a,
       to_solver_smt_ast<bitw_smt_ast>(v2)->a,
       to_solver_smt_ast<bitw_smt_ast>(v3)->a}),
    v1->sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_SQRT,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(rd)->a}),
    rd->sort);
}

smt_astt bitwuzla_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_RTI,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a}),
    from->sort);
}

smt_astt
bitwuzla_convt::mk_fp_pred(bitwuzla::Kind kind, smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    tm.mk_term(
      kind,
      {to_solver_smt_ast<bitw_smt_ast>(lhs)->a,
       to_solver_smt_ast<bitw_smt_ast>(rhs)->a}),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(bitwuzla::Kind::FP_EQUAL, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(bitwuzla::Kind::FP_GT, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(bitwuzla::Kind::FP_LT, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(bitwuzla::Kind::FP_GEQ, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(bitwuzla::Kind::FP_LEQ, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_fp_class(bitwuzla::Kind kind, smt_astt op)
{
  return new_ast(
    tm.mk_term(kind, {to_solver_smt_ast<bitw_smt_ast>(op)->a}), boolean_sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_NAN, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_INF, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_NORMAL, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_ZERO, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_NEG, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return mk_fp_class(bitwuzla::Kind::FP_IS_POS, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_ABS, {to_solver_smt_ast<bitw_smt_ast>(op)->a}),
    op->sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_neg(smt_astt op)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_NEG, {to_solver_smt_ast<bitw_smt_ast>(op)->a}),
    op->sort);
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_ubv(
  smt_astt from,
  std::size_t width)
{
  // C truncates towards zero when converting a float to an integer.
  smt_astt rm = mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_UBV,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a},
      {width}),
    mk_bv_sort(width));
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_sbv(
  smt_astt from,
  std::size_t width)
{
  smt_astt rm = mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_SBV,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a},
      {width}),
    mk_bv_sort(width));
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_FP_FROM_FP,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a},
      {to->get_exponent_width(), to->get_significand_width()}),
    to);
}

smt_astt bitwuzla_convt::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_FP_FROM_UBV,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a},
      {to->get_exponent_width(), to->get_significand_width()}),
    to);
}

smt_astt bitwuzla_convt::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_FP_FROM_SBV,
      {to_solver_smt_ast<bitw_smt_ast>(rm)->a,
       to_solver_smt_ast<bitw_smt_ast>(from)->a},
      {to->get_exponent_width(), to->get_significand_width()}),
    to);
}

smt_astt bitwuzla_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  return new_ast(
    tm.mk_term(
      bitwuzla::Kind::FP_TO_FP_FROM_BV,
      {to_solver_smt_ast<bitw_smt_ast>(op)->a},
      {to->get_exponent_width(), to->get_significand_width()}),
    to);
}

smt_astt bitwuzla_convt::mk_from_fp_to_bv(smt_astt op)
{
  /* Bitwuzla has no fp.to_ieee_bv. Mint a bit-vector symbol b and pin it with
   * op = ((_ to_fp e s) b), which the bv -> fp direction can express. The map
   * is injective away from NaN, so b is the bit pattern; for a NaN the
   * constraint holds for every NaN encoding, leaving the payload and sign
   * free (esbmc/esbmc#7021). A single shared name keeps all NaN conversions
   * agreeing with one another, as the cvc4/cvc5 backends do. */
  smt_sortt to = mk_bvfp_sort(
    op->sort->get_exponent_width(), op->sort->get_significand_width() - 1);

  const bool is_nan = to_solver_smt_ast<bitw_smt_ast>(op)->a.is_fp_value_nan();
  const std::string name =
    is_nan ? "__ESBMC_NaN"
           : "__ESBMC_to_ieeebv" + std::to_string(to_bv_counter++);

  smt_astt bv = mk_smt_symbol(name, to);
  assert_ast(mk_eq(op, mk_from_bv_to_fp(bv, op->sort)));
  return bv;
}

ieee_floatt bitwuzla_convt::get_fpbv(smt_astt a)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);

  auto [sign, exponent, significand] = guarded([&] {
    return bitw->get_value(ast->a)
      .value<std::tuple<std::string, std::string, std::string>>(2);
  });

  const unsigned ew = a->sort->get_exponent_width();
  const unsigned sw = a->sort->get_significand_width() - 1;

  ieee_floatt number(ieee_float_spect(sw, ew));
  number.unpack(binary2integer(sign + exponent + significand, false));
  return number;
}

smt_astt bitwuzla_convt::mk_quantifier(
  bool is_forall,
  std::vector<smt_astt> lhs,
  smt_astt rhs)
{
  std::vector<bitwuzla::Term> original_terms;
  std::vector<bitwuzla::Term> bound_vars;
  original_terms.reserve(lhs.size());
  bound_vars.reserve(lhs.size());

  for (size_t i = 0; i < lhs.size(); i++)
  {
    bitwuzla::Term orig = to_solver_smt_ast<bitw_smt_ast>(lhs[i])->a;
    original_terms.push_back(orig);
    std::string name =
      "qvar_" + std::to_string(quantifier_counter) + "_" + std::to_string(i);
    bound_vars.push_back(tm.mk_var(orig.sort(), name));
  }

  // Substitute SSA terms with bound vars in the body.
  // Args to mk_term: [var0, ..., varN-1, body] — no VARIABLE_LIST
  // wrapper needed (unlike CVC5).
  std::unordered_map<bitwuzla::Term, bitwuzla::Term> substitution;
  for (size_t i = 0; i < original_terms.size(); i++)
    substitution.emplace(original_terms[i], bound_vars[i]);

  bitwuzla::Term body =
    tm.substitute_term(to_solver_smt_ast<bitw_smt_ast>(rhs)->a, substitution);

  std::vector<bitwuzla::Term> args(bound_vars);
  args.push_back(body);

  return new_ast(
    tm.mk_term(
      is_forall ? bitwuzla::Kind::FORALL : bitwuzla::Kind::EXISTS, args),
    rhs->sort);
}
