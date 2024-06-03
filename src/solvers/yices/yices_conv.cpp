#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <sstream>
#include <yices_conv.h>
#include <assert.h>

// From yices 2.3 (I think) various API calls have had new non-binary
// operand versions added. The maintainers have chosen to break backwards
// compatibility in the process by moving the old functions to new names, and
// using the old names for the non-binary version. (This is a design decision).

#if __YICES_VERSION > 2 || (__YICES_VERSION == 2 && __YICES_VERSION_MAJOR >= 3)
#  define yices_bvxor yices_bvxor2
#  define yices_bvconcat yices_bvconcat2
#  define yices_bvand yices_bvand2
#  define yices_bvor yices_bvor2
#endif

#define new_ast new_solver_ast<yices_smt_ast>

smt_convt *create_new_yices_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  yices_convt *conv = new yices_convt(ns, options);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  *tuple_api = static_cast<tuple_iface *>(conv);
  return conv;
}

yices_convt::yices_convt(const namespacet &ns, const optionst &options)
  : smt_convt(ns, options), array_iface(false, false), fp_convt(this)
{
  yices_init();

  yices_clear_error();

  ctx_config_t *config = yices_new_config();
  if (options.get_bool_option("int-encoding"))
    yices_default_config_for_logic(config, "QF_AUFLIRA");
  else
    yices_default_config_for_logic(config, "QF_AUFBV");

  yices_set_config(config, "mode", "push-pop");

  yices_ctx = yices_new_context(config);
  yices_free_config(config);
}

yices_convt::~yices_convt()
{
  yices_free_context(yices_ctx);
}

void yices_convt::push_ctx()
{
  smt_convt::push_ctx();
  int32_t res = yices_push(yices_ctx);

  if (res != 0)
  {
    yices_print_error(messaget::state.out);
    log_error("Error pushing yices context");
    abort();
  }
}

void yices_convt::pop_ctx()
{
  int32_t res = yices_pop(yices_ctx);

  if (res != 0)
  {
    yices_print_error(messaget::state.out);
    log_error("Error poping yices context");
    abort();
  }

  smt_convt::pop_ctx();
}

smt_convt::resultt yices_convt::dec_solve()
{
  pre_solve();

  smt_status_t result = yices_check_context(yices_ctx, nullptr);
  if (result == STATUS_SAT)
    return smt_convt::P_SATISFIABLE;

  if (result == STATUS_UNSAT)
    return smt_convt::P_UNSATISFIABLE;

  return smt_convt::P_ERROR;
}

const std::string yices_convt::solver_text()
{
  std::stringstream ss;
  ss << "Yices version " << yices_version;
  return ss.str();
}

void yices_convt::assert_ast(smt_astt a)
{
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);
  yices_assert_formula(yices_ctx, ast->a);
}

smt_astt yices_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    yices_add(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvadd(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    yices_sub(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsub(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    yices_mul(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvmul(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_mod(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  term_t temp_term = yices_division(
    to_solver_smt_ast<yices_smt_ast>(a)->a,
    to_solver_smt_ast<yices_smt_ast>(b)->a);
  temp_term = yices_mul(temp_term, to_solver_smt_ast<yices_smt_ast>(b)->a);
  temp_term = yices_sub(to_solver_smt_ast<yices_smt_ast>(a)->a, temp_term);
  return new_ast(temp_term, a->sort);
}

smt_astt yices_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsrem(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvrem(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_div(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    yices_division(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsdiv(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvdiv(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_shl(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);

  term_t p = yices_power(
    to_solver_smt_ast<yices_smt_ast>(mk_smt_bv(2, b->sort))->a,
    to_solver_smt_ast<yices_smt_ast>(b)->a);
  return new_ast(yices_mul(to_solver_smt_ast<yices_smt_ast>(a)->a, p), a->sort);
}

smt_astt yices_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvshl(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvashr(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvlshr(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(yices_neg(to_solver_smt_ast<yices_smt_ast>(a)->a), a->sort);
}

smt_astt yices_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(yices_bvneg(to_solver_smt_ast<yices_smt_ast>(a)->a), a->sort);
}

smt_astt yices_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(yices_bvnot(to_solver_smt_ast<yices_smt_ast>(a)->a), a->sort);
}

smt_astt yices_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvxnor(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvnor(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvnand(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvxor(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvor(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvand(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    a->sort);
}

smt_astt yices_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    yices_implies(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    yices_xor2(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    yices_or2(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    yices_and2(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    yices_not(to_solver_smt_ast<yices_smt_ast>(a)->a), boolean_sort);
}

smt_astt yices_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    yices_arith_lt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvlt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvslt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    yices_arith_gt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvgt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsgt_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    yices_arith_leq_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvle_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsle_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    yices_arith_geq_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvge_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    yices_bvsge_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  if (a->sort->id == SMT_SORT_BOOL || a->sort->id == SMT_SORT_STRUCT)
    return new_ast(
      yices_eq(
        to_solver_smt_ast<yices_smt_ast>(a)->a,
        to_solver_smt_ast<yices_smt_ast>(b)->a),
      boolean_sort);

  if (int_encoding)
    return new_ast(
      yices_arith_eq_atom(
        to_solver_smt_ast<yices_smt_ast>(a)->a,
        to_solver_smt_ast<yices_smt_ast>(b)->a),
      boolean_sort);

  return new_ast(
    yices_bveq_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  if (a->sort->id == SMT_SORT_BOOL || a->sort->id == SMT_SORT_STRUCT)
    return new_ast(
      yices_neq(
        to_solver_smt_ast<yices_smt_ast>(a)->a,
        to_solver_smt_ast<yices_smt_ast>(b)->a),
      boolean_sort);

  if (int_encoding)
    return new_ast(
      yices_arith_neq_atom(
        to_solver_smt_ast<yices_smt_ast>(a)->a,
        to_solver_smt_ast<yices_smt_ast>(b)->a),
      boolean_sort);

  return new_ast(
    yices_bvneq_atom(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt yices_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());

  term_t temp_term = to_solver_smt_ast<yices_smt_ast>(b)->a;
  return new_ast(
    yices_update(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      1,
      &temp_term,
      to_solver_smt_ast<yices_smt_ast>(c)->a),
    a->sort);
}

smt_astt yices_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  term_t temp_term = to_solver_smt_ast<yices_smt_ast>(b)->a;
  return new_ast(
    yices_application(to_solver_smt_ast<yices_smt_ast>(a)->a, 1, &temp_term),
    a->sort->get_range_sort());
}

smt_astt yices_convt::mk_isint(smt_astt)
{
  log_error(
    "Yices does not support an is-integer operation on reals, "
    "therefore certain casts and operations don't work, sorry");
  abort();
}

smt_astt yices_convt::mk_smt_int(const BigInt &theint)
{
  term_t term = yices_int64(theint.to_int64());
  smt_sortt s = mk_int_sort();
  return new_ast(term, s);
}

smt_astt yices_convt::mk_smt_real(const std::string &str)
{
  term_t term = yices_parse_rational(str.c_str());
  smt_sortt s = mk_real_sort();
  return new_ast(term, s);
}

smt_astt yices_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  std::size_t w = s->get_data_width();
  term_t term;

  if (theint.is_int64())
    term = yices_bvconst_int64(w, theint.to_int64());
  else if (theint.is_uint64())
    term = yices_bvconst_uint64(w, theint.to_uint64());
  else
  {
    std::string bits = integer2binary(theint, w);
    term = yices_parse_bvbin(bits.c_str());
  }

  return new_ast(term, s);
}

smt_astt yices_convt::mk_smt_bool(bool val)
{
  smt_sortt s = boolean_sort;
  if (val)
    return new_ast(yices_true(), s);
  else
    return new_ast(yices_false(), s);
}

smt_astt yices_convt::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  // Is this term already in the symbol table?
  term_t term = yices_get_term_by_name(name.c_str());
  if (term == NULL_TERM)
  {
    // No: create a new one.
    term = yices_new_uninterpreted_term(to_solver_smt_sort<type_t>(s)->s);

    // If that wasn't the error term, set it's name.
    if (term != NULL_TERM)
      yices_set_term_name(term, name.c_str());
  }

  return new_ast(term, s);
}

smt_astt yices_convt::mk_array_symbol(
  const std::string &name,
  smt_sortt s,
  smt_sortt array_subtype [[maybe_unused]])
{
  // For array symbols, store the symbol name in the ast to implement
  // assign semantics
  const yices_smt_ast *ast =
    to_solver_smt_ast<yices_smt_ast>(mk_smt_symbol(name, s));
  const_cast<yices_smt_ast *>(ast)->symname = name;
  return ast;
}

smt_astt
yices_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);
  term_t term = yices_bvextract(ast->a, low, high);
  return new_ast(term, s);
}

smt_astt yices_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);
  term_t term = yices_sign_extend(ast->a, topwidth);
  return new_ast(term, s);
}

smt_astt yices_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);
  term_t term = yices_zero_extend(ast->a, topwidth);
  return new_ast(term, s);
}

smt_astt yices_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    yices_bvconcat(
      to_solver_smt_ast<yices_smt_ast>(a)->a,
      to_solver_smt_ast<yices_smt_ast>(b)->a),
    s);
}

smt_astt yices_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  return new_ast(
    yices_ite(
      to_solver_smt_ast<yices_smt_ast>(cond)->a,
      to_solver_smt_ast<yices_smt_ast>(t)->a,
      to_solver_smt_ast<yices_smt_ast>(f)->a),
    t->sort);
}

smt_astt
yices_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

bool yices_convt::get_bool(smt_astt a)
{
  int32_t val;
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);
  if (yices_get_bool_value(yices_get_model(yices_ctx, 1), ast->a, &val))
  {
    log_error("Can't get boolean value from Yices");
    abort();
  }

  return val ? true : false;
}

BigInt yices_convt::get_bv(smt_astt a, bool is_signed)
{
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(a);

  int64_t val = 0;
  if (int_encoding)
  {
    yices_get_int64_value(yices_get_model(yices_ctx, 1), ast->a, &val);
    return BigInt(val);
  }

  unsigned int width = a->sort->get_data_width();

  int32_t *data = new int32_t[width];
  yices_get_bv_value(yices_get_model(yices_ctx, 1), ast->a, data);

  std::string s;
  for (unsigned i = 0; i < width; i++)
    s.append(std::to_string(data[width - i - 1]));

  delete[] data;

  return binary2integer(s, is_signed);
}

expr2tc yices_convt::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  // Construct a term accessing that element, and get_bv it.
  const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(array);
  term_t idx;
  if (int_encoding)
  {
    idx = yices_int64(index);
  }
  else
  {
    idx = yices_bvconst_uint64(array->sort->get_domain_width(), index);
  }

  term_t app = yices_application(ast->a, 1, &idx);
  smt_sortt subsort = convert_sort(subtype);
  smt_astt container = new_ast(app, subsort);
  return get_by_ast(subtype, container);
}

expr2tc yices_convt::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  return get_array_elem(array, index, get_flattened_array_subtype(subtype));
}

void yices_smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  if (sort->id == SMT_SORT_ARRAY)
  {
    // Perform assign semantics, of this to the given sym
    const yices_smt_ast *ast = to_solver_smt_ast<yices_smt_ast>(sym);
    yices_remove_term_name(ast->symname.c_str());
    yices_set_term_name(a, ast->symname.c_str());

    // set the other ast too
    const_cast<yices_smt_ast *>(ast)->a = a;
  }
  else
  {
    smt_ast::assign(ctx, sym);
  }
}

smt_astt yices_smt_ast::project(smt_convt *ctx, unsigned int elem) const
{
  type2tc type = sort->get_tuple_type();
  const struct_union_data &data = ctx->get_type_def(type);
  smt_sortt elemsort = ctx->convert_sort(data.members[elem]);

  return ctx->new_ast(yices_select(elem + 1, a), elemsort);
}

smt_astt yices_smt_ast::update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  if (sort->id == SMT_SORT_ARRAY)
    return smt_ast::update(ctx, value, idx, idx_expr);

  // Otherwise, it's a struct
  assert(sort->id == SMT_SORT_STRUCT);
  assert(is_nil_expr(idx_expr) && "Tuple updates must be explicitly numbered");

  const yices_smt_ast *yast = to_solver_smt_ast<yices_smt_ast>(value);
  term_t result = yices_tuple_update(a, idx + 1, yast->a);
  return ctx->new_ast(result, sort);
}

smt_sortt yices_convt::mk_struct_sort(const type2tc &type)
{
  // Exactly the same as a normal yices sort, ish.

  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    smt_sortt subtypesort = convert_sort(arrtype.subtype);
    smt_sortt d = mk_int_bv_sort(make_array_domain_type(arrtype)->get_width());
    return mk_array_sort(d, subtypesort);
  }

  std::vector<type_t> sorts;
  const struct_union_data &def = get_type_def(type);
  for (auto const &it : def.members)
  {
    smt_sortt s = convert_sort(it);
    sorts.push_back(to_solver_smt_sort<type_t>(s)->s);
  }

  // We now have an array of types, ready for sort creation
  type_t tuple_sort = yices_tuple_type(def.members.size(), sorts.data());
  return new solver_smt_sort<type_t>(SMT_SORT_STRUCT, tuple_sort, type);
}

smt_astt yices_convt::tuple_create(const expr2tc &structdef)
{
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type = get_type_def(strct.type);

  std::vector<term_t> terms;
  for (auto const &it : strct.datatype_members)
  {
    smt_astt a = convert_ast(it);
    const yices_smt_ast *yast = to_solver_smt_ast<yices_smt_ast>(a);
    terms.push_back(yast->a);
  }

  term_t thetuple = yices_tuple(type.members.size(), terms.data());
  return new_ast(thetuple, convert_sort(strct.type));
}

smt_astt yices_convt::tuple_fresh(smt_sortt s, std::string name)
{
  term_t t = yices_new_uninterpreted_term(to_solver_smt_sort<type_t>(s)->s);
  yices_set_term_name(t, name.c_str());
  return new_ast(t, s);
}

smt_astt yices_convt::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain [[maybe_unused]])
{
  const array_type2t &arr_type = to_array_type(array_type);
  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  unsigned int sz = thesize.value.to_uint64();

  // We support both tuples and arrays of them, so just repeatedly store
  smt_sortt sort = convert_sort(array_type);
  std::string name = mk_fresh_name("yices_convt::tuple_array_create");
  smt_astt a = tuple_fresh(sort, name);

  if (const_array)
  {
    smt_astt init = inputargs[0];
    for (unsigned int i = 0; i < sz; i++)
    {
      a = a->update(this, init, i);
    }

    return a;
  }
  else
  {
    // Repeatedly store operands into this.
    for (unsigned int i = 0; i < sz; i++)
    {
      a = a->update(this, inputargs[i], i);
    }

    return a;
  }
}

smt_astt yices_convt::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  smt_sortt subs = convert_sort(init_value->type);

  type2tc domtype = unsignedbv_type2tc(domain_width);
  smt_sortt doms = convert_sort(domtype);

  type_t tuplearr = yices_function_type(
    1,
    &to_solver_smt_sort<type_t>(doms)->s,
    to_solver_smt_sort<type_t>(subs)->s);
  term_t theterm = yices_new_uninterpreted_term(tuplearr);

  smt_astt a = convert_ast(init_value);
  const yices_smt_ast *yast = to_solver_smt_ast<yices_smt_ast>(a);

  // Now repeatedly store Things into it
  unsigned int elems =
    to_constant_int2t(array_domain_to_width(domtype)).value.to_uint64();
  for (unsigned int i = 0; i < elems; i++)
  {
    term_t idxterm =
      int_encoding ? yices_int64(i) : yices_bvconst_uint64(domain_width, i);

    theterm = yices_update(theterm, 1, &idxterm, yast->a);
  }

  smt_sortt retsort = new solver_smt_sort<type_t>(SMT_SORT_STRUCT, tuplearr);
  return new_ast(theterm, retsort);
}

smt_astt yices_convt::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_astt yices_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
}

expr2tc yices_convt::tuple_get(const type2tc &type, smt_astt sym)
{
  const struct_union_data &strct = get_type_def(type);

  if (is_pointer_type(type))
  {
    // Pointer have two fields, a base address and an offset, so we just
    // need to get the two numbers and call the pointer API

    auto s1 = convert_sort(strct.members[0]);
    smt_astt object =
      new_ast(yices_select(1, to_solver_smt_ast<yices_smt_ast>(sym)->a), s1);

    auto s2 = convert_sort(strct.members[1]);
    smt_astt offset =
      new_ast(yices_select(2, to_solver_smt_ast<yices_smt_ast>(sym)->a), s2);

    unsigned int num =
      get_bv(object, is_signedbv_type(strct.members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(strct.members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, type);
  }

  // Otherwise, run through all fields and despatch to 'get_by_ast' again.
  std::vector<expr2tc> outmem;
  unsigned int i = 0;
  for (auto const &it : strct.members)
  {
    outmem.push_back(smt_convt::get_by_ast(
      it,
      new_ast(
        yices_select(1 + i, to_solver_smt_ast<yices_smt_ast>(sym)->a),
        convert_sort(it))));
    i++;
  }

  return constant_struct2tc(type, std::move(outmem));
}

expr2tc yices_convt::tuple_get(const expr2tc &expr)
{
  const struct_union_data &strct = get_type_def(expr->type);

  if (is_pointer_type(expr->type))
  {
    // Pointer have two fields, a base address and an offset, so we just
    // need to get the two numbers and call the pointer API

    smt_astt sym = convert_ast(expr);

    auto s1 = convert_sort(strct.members[0]);
    smt_astt object =
      new_ast(yices_select(1, to_solver_smt_ast<yices_smt_ast>(sym)->a), s1);

    auto s2 = convert_sort(strct.members[1]);
    smt_astt offset =
      new_ast(yices_select(2, to_solver_smt_ast<yices_smt_ast>(sym)->a), s2);

    unsigned int num =
      get_bv(object, is_signedbv_type(strct.members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(strct.members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  // Otherwise, run through all fields and despatch to 'get' again.
  std::vector<expr2tc> outmem;
  unsigned int i = 0;
  for (auto const &it : strct.members)
  {
    expr2tc memb = member2tc(it, expr, strct.member_names[i]);
    outmem.push_back(get(memb));
    i++;
  }

  return constant_struct2tc(expr->type, std::move(outmem));
}

void yices_convt::print_model()
{
  yices_print_model(messaget::state.out, yices_get_model(yices_ctx, 1));
}

smt_sortt yices_convt::mk_bool_sort()
{
  return new solver_smt_sort<type_t>(SMT_SORT_BOOL, yices_bool_type(), 1);
}

smt_sortt yices_convt::mk_real_sort()
{
  return new solver_smt_sort<type_t>(SMT_SORT_REAL, yices_int_type());
}

smt_sortt yices_convt::mk_int_sort()
{
  return new solver_smt_sort<type_t>(SMT_SORT_INT, yices_real_type());
}

smt_sortt yices_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<type_t>(SMT_SORT_BV, yices_bv_type(width), width);
}

smt_sortt yices_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<type_t>(
    SMT_SORT_FIXEDBV, yices_bv_type(width), width);
}

smt_sortt yices_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<type_t>(domain);
  auto range_sort = to_solver_smt_sort<type_t>(range);

  auto t = yices_function_type(1, &domain_sort->s, range_sort->s);
  return new solver_smt_sort<type_t>(
    SMT_SORT_ARRAY, t, domain_sort->get_data_width(), range);
}

smt_sortt yices_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<type_t>(
    SMT_SORT_BVFP, yices_bv_type(ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt yices_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<type_t>(SMT_SORT_BVFP_RM, yices_bv_type(3), 3);
}

void yices_smt_ast::dump() const
{
  yices_pp_term(messaget::state.out, a, 80, 10, 0);
  yices_pp_type(
    messaget::state.out, to_solver_smt_sort<type_t>(sort)->s, 80, 10, 0);
}
