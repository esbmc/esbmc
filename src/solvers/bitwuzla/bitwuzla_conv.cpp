#include <bitwuzla_conv.h>
#include <cstdio>

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
/** Owns the term reference bitwuzla_get_value() hands back. The C API refcounts
 *  every term it exports, so a model query that never releases its result pins
 *  that term in the term manager for the lifetime of the solver.
 *
 *  Balanced because every export is exactly +1: releasing once undoes the
 *  query's increment and no more. Where the query hands back the argument
 *  itself (issue #7063) that still leaves ESBMC's own reference, taken when the
 *  term was built, intact -- this holds only while nothing else in this file
 *  releases a term ESBMC still stores. */
class value_reft
{
public:
  explicit value_reft(BitwuzlaTerm t) : t(t)
  {
  }
  ~value_reft()
  {
    bitwuzla_term_release(t);
  }
  value_reft(const value_reft &) = delete;
  value_reft &operator=(const value_reft &) = delete;

  operator BitwuzlaTerm() const
  {
    return t;
  }

private:
  BitwuzlaTerm t;
};
} // namespace

void bitwuzla_error_handler(const char *msg)
{
  log_error("Bitwuzla error encountered\n{}", msg);
  abort();
}

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

  bitw_options = bitwuzla_options_new();
  bitw_term_manager = bitwuzla_term_manager_new();
  bitwuzla_set_option(bitw_options, BITWUZLA_OPT_PRODUCE_MODELS, 1);
  bitwuzla_set_abort_callback(bitwuzla_error_handler);
  bitw = bitwuzla_new(bitw_term_manager, bitw_options);
}

bitwuzla_convt::~bitwuzla_convt()
{
  bitwuzla_delete(bitw);
  bitwuzla_options_delete(bitw_options);
  bitwuzla_term_manager_delete(bitw_term_manager);
  bitw = nullptr;
  bitw_options = nullptr;
}

void bitwuzla_convt::push_ctx()
{
  smt_solver_baset::push_ctx();
  bitwuzla_push(bitw, 1);
}

void bitwuzla_convt::pop_ctx()
{
  symtabt::nth_index<1>::type &symtab_levels = symtable.get<1>();
  symtab_levels.erase(ctx_level);

  bitwuzla_pop(bitw, 1);
  smt_solver_baset::pop_ctx();
}

smt_resultt bitwuzla_convt::dec_solve()
{
  pre_solve();

  BitwuzlaResult result = bitwuzla_check_sat(bitw);

  if (result == BITWUZLA_SAT)
    return P_SATISFIABLE;

  if (result == BITWUZLA_UNSAT)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string bitwuzla_convt::solver_text()
{
  std::string ss = "Bitwuzla ";
  ss += bitwuzla_version();
  return ss;
}

void bitwuzla_convt::assert_ast(smt_astt a)
{
  bitwuzla_assert(bitw, to_solver_smt_ast<bitw_smt_ast>(a)->a);
}

smt_astt bitwuzla_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_ADD,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SUB,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_MUL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SREM,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_UREM,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SDIV,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_UDIV,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SHL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_ASHR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SHR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager,
      BITWUZLA_KIND_BV_NEG,
      to_solver_smt_ast<bitw_smt_ast>(a)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager,
      BITWUZLA_KIND_BV_NOT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_XOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_OR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_AND,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_IMPLIES,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_XOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_OR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_AND,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager,
      BITWUZLA_KIND_NOT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_ULT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SLT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_UGT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SGT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_ULE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SLE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_UGE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_SGE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_EQUAL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_DISTINCT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term3(
      bitw_term_manager,
      BITWUZLA_KIND_ARRAY_STORE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a,
      to_solver_smt_ast<bitw_smt_ast>(c)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_ARRAY_SELECT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
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
    bitwuzla_mk_bv_value(
      bitw_term_manager,
      to_solver_smt_sort<BitwuzlaSort>(s)->s,
      integer2binary(theint, s->get_data_width()).c_str(),
      2),
    s);
}

smt_astt bitwuzla_convt::mk_smt_bool(bool val)
{
  BitwuzlaTerm node = (val) ? bitwuzla_mk_true(bitw_term_manager)
                            : bitwuzla_mk_false(bitw_term_manager);
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

  BitwuzlaTerm node;

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
    node = bitwuzla_mk_const(
      bitw_term_manager, to_solver_smt_sort<BitwuzlaSort>(s)->s, name.c_str());
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
  // every bitwuzla_mk_const, so it is cached and reused across applications;
  // sharing one declaration is what makes the solver enforce congruence.
  auto it = uf_decls.find(name);
  BitwuzlaTerm fun;
  if (it != uf_decls.end())
    fun = it->second;
  else
  {
    std::vector<BitwuzlaSort> domain;
    domain.reserve(args.size());
    for (smt_astt arg : args)
      domain.push_back(to_solver_smt_sort<BitwuzlaSort>(arg->sort)->s);

    BitwuzlaSort fun_sort = bitwuzla_mk_fun_sort(
      bitw_term_manager,
      domain.size(),
      domain.data(),
      to_solver_smt_sort<BitwuzlaSort>(rangesort)->s);
    fun = bitwuzla_mk_const(bitw_term_manager, fun_sort, name.c_str());
    uf_decls.emplace(name, fun);
  }

  // BITWUZLA_KIND_APPLY expects [function, arg0, arg1, ...].
  std::vector<BitwuzlaTerm> apply_args;
  apply_args.reserve(args.size() + 1);
  apply_args.push_back(fun);
  for (smt_astt arg : args)
    apply_args.push_back(to_solver_smt_ast<bitw_smt_ast>(arg)->a);

  return new_ast(
    bitwuzla_mk_term(
      bitw_term_manager,
      BITWUZLA_KIND_APPLY,
      apply_args.size(),
      apply_args.data()),
    rangesort);
}

smt_astt
bitwuzla_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  BitwuzlaTerm b = bitwuzla_mk_term1_indexed2(
    bitw_term_manager, BITWUZLA_KIND_BV_EXTRACT, ast->a, high, low);
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  BitwuzlaTerm b = bitwuzla_mk_term1_indexed1(
    bitw_term_manager, BITWUZLA_KIND_BV_SIGN_EXTEND, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  BitwuzlaTerm b = bitwuzla_mk_term1_indexed1(
    bitw_term_manager, BITWUZLA_KIND_BV_ZERO_EXTEND, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_BV_CONCAT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
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
    bitwuzla_mk_term3(
      bitw_term_manager,
      BITWUZLA_KIND_ITE,
      to_solver_smt_ast<bitw_smt_ast>(cond)->a,
      to_solver_smt_ast<bitw_smt_ast>(t)->a,
      to_solver_smt_ast<bitw_smt_ast>(f)->a),
    t->sort);
}

tvt bitwuzla_convt::get_bool(smt_astt a)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  value_reft value(bitwuzla_get_value(bitw, ast->a));

  if (bitwuzla_term_is_true(value))
    return tvt(true);
  if (bitwuzla_term_is_false(value))
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
  value_reft value(bitwuzla_get_value(bitw, ast->a));
  return binary2integer(bitwuzla_term_value_get_str(value), is_signed);
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

  value_reft array_value(bitwuzla_get_value(bitw, za->a));
  value_reft index_value(bitwuzla_get_value(bitw, idx->a));

  BitwuzlaTerm e = bitwuzla_mk_term2(
    bitw_term_manager, BITWUZLA_KIND_ARRAY_SELECT, array_value, index_value);

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

  BitwuzlaTerm res;
  if (is_add2t(overflow.operand))
  {
    if (is_signed)
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_SADD_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_UADD_OVERFLOW, side1->a, side2->a);
    }
  }
  else if (is_sub2t(overflow.operand))
  {
    if (is_signed)
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_SSUB_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_USUB_OVERFLOW, side1->a, side2->a);
    }
  }
  else if (is_mul2t(overflow.operand))
  {
    if (is_signed)
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_SMUL_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      res = bitwuzla_mk_term2(
        bitw_term_manager, BITWUZLA_KIND_BV_UMUL_OVERFLOW, side1->a, side2->a);
    }
  }
  else if (is_div2t(overflow.operand) || is_modulus2t(overflow.operand))
  {
    res = bitwuzla_mk_term2(
      bitw_term_manager, BITWUZLA_KIND_BV_SDIV_OVERFLOW, side1->a, side2->a);
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
    bitwuzla_mk_const_array(
      bitw_term_manager,
      to_solver_smt_sort<BitwuzlaSort>(arrsort)->s,
      to_solver_smt_ast<bitw_smt_ast>(init_val)->a),
    arrsort);
}

std::string bitwuzla_convt::dump_smt()
{
  FILE *temp_file = tmpfile();
  if (!temp_file)
  {
    log_error("Failed to create temporary file for SMT dump");
    return "";
  }

  bitwuzla_print_formula(bitw, "smt2", temp_file, 2);

  // Get file size and read entire content
  fseek(temp_file, 0, SEEK_END);
  long file_size = ftell(temp_file);
  fseek(temp_file, 0, SEEK_SET);

  if (file_size <= 0)
  {
    fclose(temp_file);
    return "";
  }

  // Allocate buffer for entire file content
  std::vector<char> buffer(file_size + 1);
  size_t bytes_read = fread(buffer.data(), 1, file_size, temp_file);
  buffer[bytes_read] = '\0'; // Null terminate

  fclose(temp_file);
  return std::string(buffer.data(), bytes_read);
}

void bitw_smt_ast::dump() const
{
  log_status("{}", bitwuzla_term_to_string(a));
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
    smt_astt term = entry.ast;
    BitwuzlaSort sort =
      bitwuzla_term_get_sort(to_solver_smt_ast<bitw_smt_ast>(term)->a);
    fprintf(
      messaget::state.out,
      "(define-fun %s (",
      bitwuzla_term_get_symbol(to_solver_smt_ast<bitw_smt_ast>(term)->a));
    if (bitwuzla_sort_is_fun(sort))
    {
      BitwuzlaTerm value =
        bitwuzla_get_value(bitw, to_solver_smt_ast<bitw_smt_ast>(term)->a);
      /* value walks down the lambda chain below; this owns the reference the
       * query handed back, which the walk leaves behind on its first step. The
       * walk stays valid past that release because bitwuzla_term_get_children
       * exports every child with its own increment and never drops it. */
      value_reft root(value);
      size_t size;
      BitwuzlaTerm *children = bitwuzla_term_get_children(value, &size);
      assert(size == 2);
      while (bitwuzla_term_get_kind(children[1]) == BITWUZLA_KIND_LAMBDA)
      {
        assert(bitwuzla_term_is_var(children[0]));
        fprintf(
          messaget::state.out,
          "(%s %s) ",
          bitwuzla_term_to_string(children[0]),
          bitwuzla_sort_to_string(bitwuzla_term_get_sort(children[0])));
        value = children[1];
        children = bitwuzla_term_get_children(value, &size);
      }
      assert(bitwuzla_term_is_var(children[0]));
      // Note: The returned string of bitwuzla_term_to_string and
      //       bitwuzla_sort_to_string does not have to be freed, but is only
      //       valid until the next call to the respective function. Thus we
      //       split printing into separate printf calls so that none of these
      //       functions is called more than once in one printf call.
      //       Alternatively, we could also first get and copy the strings, use
      //       a single printf call, and then free the copied strings.
      fprintf(
        messaget::state.out,
        "(%s %s))",
        bitwuzla_term_to_string(children[0]),
        bitwuzla_sort_to_string(bitwuzla_term_get_sort(children[0])));
      fprintf(
        messaget::state.out,
        " %s",
        bitwuzla_sort_to_string(bitwuzla_sort_fun_get_codomain(sort)));
      fprintf(
        messaget::state.out, " %s)\n", bitwuzla_term_to_string(children[1]));
    }
    else
    {
      value_reft value(
        bitwuzla_get_value(bitw, to_solver_smt_ast<bitw_smt_ast>(term)->a));
      fprintf(
        messaget::state.out,
        ") %s %s)\n",
        bitwuzla_sort_to_string(sort),
        bitwuzla_term_to_string(value));
    }
  }
}

smt_sortt bitwuzla_convt::mk_bool_sort()
{
  return cached_sort({SMT_SORT_BOOL, 0, 0}, [this] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_BOOL, bitwuzla_mk_bool_sort(bitw_term_manager), 1);
  });
}

smt_sortt bitwuzla_convt::mk_bv_sort(std::size_t width)
{
  return cached_sort({SMT_SORT_BV, width, 0}, [this, width] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_BV, bitwuzla_mk_bv_sort(bitw_term_manager, width), width);
  });
}

smt_sortt bitwuzla_convt::mk_fbv_sort(std::size_t width)
{
  return cached_sort({SMT_SORT_FIXEDBV, width, 0}, [this, width] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_FIXEDBV, bitwuzla_mk_bv_sort(bitw_term_manager, width), width);
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
    auto domain_sort = to_solver_smt_sort<BitwuzlaSort>(domain);
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_ARRAY,
      bitwuzla_mk_array_sort(
        bitw_term_manager,
        domain_sort->s,
        to_solver_smt_sort<BitwuzlaSort>(range)->s),
      domain_sort->get_data_width(),
      range);
  });
}

smt_sortt bitwuzla_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return cached_sort({SMT_SORT_BVFP, ew, sw}, [this, ew, sw] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_BVFP,
      bitwuzla_mk_bv_sort(bitw_term_manager, ew + sw + 1),
      ew + sw + 1,
      sw + 1);
  });
}

smt_sortt bitwuzla_convt::mk_bvfp_rm_sort()
{
  return cached_sort({SMT_SORT_BVFP_RM, 0, 0}, [this] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_BVFP_RM, bitwuzla_mk_bv_sort(bitw_term_manager, 3), 3);
  });
}

smt_sortt bitwuzla_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  // sw excludes the hidden bit, which Bitwuzla's significand width includes.
  return cached_sort({SMT_SORT_FPBV, ew, sw}, [this, ew, sw] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_FPBV,
      bitwuzla_mk_fp_sort(bitw_term_manager, ew, sw + 1),
      ew + sw + 1,
      sw + 1);
  });
}

smt_sortt bitwuzla_convt::mk_fpbv_rm_sort()
{
  return cached_sort({SMT_SORT_FPBV_RM, 0, 0}, [this] {
    return new solver_smt_sort<BitwuzlaSort>(
      SMT_SORT_FPBV_RM, bitwuzla_mk_rm_sort(bitw_term_manager), 3);
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
  return new_ast(
    bitwuzla_mk_fp_nan(
      bitw_term_manager, to_solver_smt_sort<BitwuzlaSort>(s)->s),
    s);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_fpbv_sort(ew, sw - 1);
  BitwuzlaSort bs = to_solver_smt_sort<BitwuzlaSort>(s)->s;
  return new_ast(
    sgn ? bitwuzla_mk_fp_neg_inf(bitw_term_manager, bs)
        : bitwuzla_mk_fp_pos_inf(bitw_term_manager, bs),
    s);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  BitwuzlaRoundingMode brm;
  switch (rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    brm = BITWUZLA_RM_RNE;
    break;
  case ieee_floatt::ROUND_TO_AWAY:
    brm = BITWUZLA_RM_RNA;
    break;
  case ieee_floatt::ROUND_TO_PLUS_INF:
    brm = BITWUZLA_RM_RTP;
    break;
  case ieee_floatt::ROUND_TO_MINUS_INF:
    brm = BITWUZLA_RM_RTN;
    break;
  case ieee_floatt::ROUND_TO_ZERO:
    brm = BITWUZLA_RM_RTZ;
    break;
  default:
    log_error("Unexpected rounding mode reached Bitwuzla");
    abort();
  }

  return new_ast(
    bitwuzla_mk_rm_value(bitw_term_manager, brm), mk_fpbv_rm_sort());
}

smt_astt bitwuzla_convt::mk_fp_arith(
  BitwuzlaKind kind,
  smt_astt lhs,
  smt_astt rhs,
  smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term3(
      bitw_term_manager,
      kind,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(lhs)->a,
      to_solver_smt_ast<bitw_smt_ast>(rhs)->a),
    lhs->sort);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(BITWUZLA_KIND_FP_ADD, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(BITWUZLA_KIND_FP_SUB, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(BITWUZLA_KIND_FP_MUL, lhs, rhs, rm);
}

smt_astt
bitwuzla_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return mk_fp_arith(BITWUZLA_KIND_FP_DIV, lhs, rhs, rm);
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
  BitwuzlaTerm args[4] = {
    to_solver_smt_ast<bitw_smt_ast>(rm)->a,
    to_solver_smt_ast<bitw_smt_ast>(v1)->a,
    to_solver_smt_ast<bitw_smt_ast>(v2)->a,
    to_solver_smt_ast<bitw_smt_ast>(v3)->a};
  return new_ast(
    bitwuzla_mk_term(bitw_term_manager, BITWUZLA_KIND_FP_FMA, 4, args),
    v1->sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_SQRT,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(rd)->a),
    rd->sort);
}

smt_astt bitwuzla_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_RTI,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a),
    from->sort);
}

smt_astt
bitwuzla_convt::mk_fp_pred(BitwuzlaKind kind, smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    bitwuzla_mk_term2(
      bitw_term_manager,
      kind,
      to_solver_smt_ast<bitw_smt_ast>(lhs)->a,
      to_solver_smt_ast<bitw_smt_ast>(rhs)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(BITWUZLA_KIND_FP_EQUAL, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(BITWUZLA_KIND_FP_GT, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(BITWUZLA_KIND_FP_LT, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(BITWUZLA_KIND_FP_GEQ, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  return mk_fp_pred(BITWUZLA_KIND_FP_LEQ, lhs, rhs);
}

smt_astt bitwuzla_convt::mk_fp_class(BitwuzlaKind kind, smt_astt op)
{
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager, kind, to_solver_smt_ast<bitw_smt_ast>(op)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_NAN, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_INF, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_NORMAL, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_ZERO, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_NEG, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return mk_fp_class(BITWUZLA_KIND_FP_IS_POS, op);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager,
      BITWUZLA_KIND_FP_ABS,
      to_solver_smt_ast<bitw_smt_ast>(op)->a),
    op->sort);
}

smt_astt bitwuzla_convt::mk_smt_fpbv_neg(smt_astt op)
{
  return new_ast(
    bitwuzla_mk_term1(
      bitw_term_manager,
      BITWUZLA_KIND_FP_NEG,
      to_solver_smt_ast<bitw_smt_ast>(op)->a),
    op->sort);
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_ubv(
  smt_astt from,
  std::size_t width)
{
  // C truncates towards zero when converting a float to an integer.
  smt_astt rm = mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  return new_ast(
    bitwuzla_mk_term2_indexed1(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_UBV,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a,
      width),
    mk_bv_sort(width));
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_sbv(
  smt_astt from,
  std::size_t width)
{
  smt_astt rm = mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);
  return new_ast(
    bitwuzla_mk_term2_indexed1(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_SBV,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a,
      width),
    mk_bv_sort(width));
}

smt_astt bitwuzla_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term2_indexed2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_FP_FROM_FP,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a,
      to->get_exponent_width(),
      to->get_significand_width()),
    to);
}

smt_astt bitwuzla_convt::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term2_indexed2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_FP_FROM_UBV,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a,
      to->get_exponent_width(),
      to->get_significand_width()),
    to);
}

smt_astt bitwuzla_convt::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return new_ast(
    bitwuzla_mk_term2_indexed2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_FP_FROM_SBV,
      to_solver_smt_ast<bitw_smt_ast>(rm)->a,
      to_solver_smt_ast<bitw_smt_ast>(from)->a,
      to->get_exponent_width(),
      to->get_significand_width()),
    to);
}

smt_astt bitwuzla_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  return new_ast(
    bitwuzla_mk_term1_indexed2(
      bitw_term_manager,
      BITWUZLA_KIND_FP_TO_FP_FROM_BV,
      to_solver_smt_ast<bitw_smt_ast>(op)->a,
      to->get_exponent_width(),
      to->get_significand_width()),
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

  const bool is_nan =
    bitwuzla_term_is_fp_value_nan(to_solver_smt_ast<bitw_smt_ast>(op)->a);
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

  const char *sign;
  const char *exponent;
  const char *significand;
  value_reft value(bitwuzla_get_value(bitw, ast->a));
  bitwuzla_term_value_get_fp_ieee(value, &sign, &exponent, &significand, 2);

  const unsigned ew = a->sort->get_exponent_width();
  const unsigned sw = a->sort->get_significand_width() - 1;

  ieee_floatt number(ieee_float_spect(sw, ew));
  number.unpack(binary2integer(
    std::string(sign) + std::string(exponent) + std::string(significand),
    false));
  return number;
}

smt_astt bitwuzla_convt::mk_quantifier(
  bool is_forall,
  std::vector<smt_astt> lhs,
  smt_astt rhs)
{
  std::vector<BitwuzlaTerm> original_terms;
  std::vector<BitwuzlaTerm> bound_vars;
  original_terms.reserve(lhs.size());
  bound_vars.reserve(lhs.size());

  for (size_t i = 0; i < lhs.size(); i++)
  {
    BitwuzlaTerm orig = to_solver_smt_ast<bitw_smt_ast>(lhs[i])->a;
    original_terms.push_back(orig);
    std::string name =
      "qvar_" + std::to_string(quantifier_counter) + "_" + std::to_string(i);
    bound_vars.push_back(bitwuzla_mk_var(
      bitw_term_manager, bitwuzla_term_get_sort(orig), name.c_str()));
  }

  // Substitute SSA terms with bound vars in the body.
  // Args to bitwuzla_mk_term: [var0, ..., varN-1, body] — no VARIABLE_LIST
  // wrapper needed (unlike CVC5).
  BitwuzlaTerm body = bitwuzla_substitute_term(
    to_solver_smt_ast<bitw_smt_ast>(rhs)->a,
    original_terms.size(),
    original_terms.data(),
    bound_vars.data());

  std::vector<BitwuzlaTerm> args(bound_vars);
  args.push_back(body);

  return new_ast(
    bitwuzla_mk_term(
      bitw_term_manager,
      is_forall ? BITWUZLA_KIND_FORALL : BITWUZLA_KIND_EXISTS,
      static_cast<uint32_t>(args.size()),
      args.data()),
    rhs->sort);
}
