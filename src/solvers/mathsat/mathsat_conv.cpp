#include <cassert>
#include <gmp.h>
#include <mathsat_conv.h>
#include <sstream>
#include <string>
#include <util/c_types.h>
#include <util/expr_util.h>

static const char *mathsat_config =
  "preprocessor.toplevel_propagation = true\n"
  "preprocessor.simplification = 1\n"
  "dpll.branching_random_frequency = 0.01\n"
  "dpll.branching_random_invalidate_phase_cache = true\n"
  "dpll.restart_strategy = 3\n"
  "dpll.glucose_var_activity = true\n"
  "dpll.glucose_learnt_minimization = true\n"
  "dpll.preprocessor.mode = 1\n"
  "theory.bv.eager = true\n"
  "theory.bv.bit_blast_mode = 2\n"
  "theory.bv.delay_propagated_eqs = true\n"
  "theory.la.enabled = false\n"
  "theory.fp.mode = 1\n"
  "theory.fp.bit_blast_mode = 2\n"
  "theory.fp.bv_combination_enabled = true\n"
  "theory.arr.enable_witness = true";

void mathsat_convt::check_msat_error(msat_term &r)
{
  if(MSAT_ERROR_TERM(r))
  {
    std::cerr << "Error creating SMT " << std::endl;
    std::cerr << "Error text: \"" << msat_last_error_message(env) << "\""
              << std::endl;
    abort();
  }
}

smt_convt *create_new_mathsat_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api __attribute__((unused)),
  array_iface **array_api,
  fp_convt **fp_api)
{
  mathsat_convt *conv = new mathsat_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

mathsat_convt::mathsat_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns),
    array_iface(false, false),
    fp_convt(this),
    use_fp_api(false)
{
  cfg = msat_parse_config(mathsat_config);
  msat_set_option(cfg, "model_generation", "true");
  env = msat_create_env(cfg);
}

mathsat_convt::~mathsat_convt()
{
  msat_destroy_env(env);
}

void mathsat_convt::push_ctx()
{
  smt_convt::push_ctx();
  msat_push_backtrack_point(env);
}

void mathsat_convt::pop_ctx()
{
  msat_pop_backtrack_point(env);
  smt_convt::pop_ctx();
}

void mathsat_convt::assert_ast(const smt_ast *a)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);
  msat_assert_formula(env, mast->a);
}

smt_convt::resultt mathsat_convt::dec_solve()
{
  pre_solve();

  msat_result r = msat_solve(env);
  if(r == MSAT_SAT)
    return P_SATISFIABLE;

  if(r == MSAT_UNSAT)
    return P_UNSATISFIABLE;

  return smt_convt::P_ERROR;
}

bool mathsat_convt::get_bool(const smt_ast *a)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);
  msat_term t = msat_get_model_value(env, mast->a);
  check_msat_error(t);

  bool res;
  if(msat_term_is_true(env, t))
    res = true;
  else if(msat_term_is_false(env, t))
    res = false;
  else
  {
    std::cerr << "Boolean model value is neither true or false" << std::endl;
    abort();
  }

  msat_free(msat_term_repr(t));
  return res;
}

BigInt mathsat_convt::get_bv(smt_astt a)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);
  msat_term t = msat_get_model_value(env, mast->a);
  check_msat_error(t);

  // GMP rational value object.
  mpq_t val;
  mpq_init(val);

  msat_term_to_number(env, t, val);
  check_msat_error(t);
  msat_free(msat_term_repr(t));

  mpz_t num;
  mpz_init(num);
  mpz_set(num, mpq_numref(val));
  char buffer[mpz_sizeinbase(num, 10) + 2];
  mpz_get_str(buffer, 10, num);

  char *foo = buffer;
  int64_t finval = strtoll(buffer, &foo, 10);

  if(buffer[0] != '\0' && (foo == buffer || *foo != '\0'))
  {
    std::cerr << "Couldn't parse string representation of number \"" << buffer
              << "\"" << std::endl;
    abort();
  }

  return BigInt(finval);
}

ieee_floatt mathsat_convt::get_fpbv(smt_astt a)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);
  msat_term t = msat_get_model_value(env, mast->a);
  check_msat_error(t);

  // GMP rational value object.
  mpq_t val;
  mpq_init(val);

  msat_term_to_number(env, t, val);
  check_msat_error(t);
  msat_free(msat_term_repr(t));

  mpz_t num;
  mpz_init(num);
  mpz_set(num, mpq_numref(val));
  char buffer[mpz_sizeinbase(num, 10) + 2];
  mpz_get_str(buffer, 10, num);

  size_t ew, sw;
  if(msat_is_fp_type(env, to_solver_smt_sort<msat_type>(a->sort)->s, &ew, &sw) != 0)
  {
    std::cerr << "Non FP type passed to mathsat_convt::get_exp_width" << std::endl;
    abort();
  }

  ieee_floatt number(ieee_float_spect(sw, ew));
  number.unpack(BigInt(buffer));
  return number;
}

expr2tc mathsat_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  size_t orig_w = array->sort->get_domain_width();
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(array);

  smt_astt tmpast = mk_smt_bv(BigInt(index), mk_bv_sort(orig_w));
  const mathsat_smt_ast *tmpa = to_solver_smt_ast<mathsat_smt_ast>(tmpast);

  msat_term t = msat_make_array_read(env, mast->a, tmpa->a);
  check_msat_error(t);

  expr2tc result = get_by_ast(subtype, new_ast(t, convert_sort(subtype)));

  msat_free(msat_term_repr(t));

  return result;
}

const std::string mathsat_convt::solver_text()
{
  std::stringstream ss;
  char *tmp = msat_get_version();
  ss << tmp;
  msat_free(tmp);
  return ss.str();
}

smt_astt mathsat_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    msat_make_plus(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_plus(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  msat_term neg_b = msat_make_times(
    env, msat_make_number(env, "-1"), to_solver_smt_ast<mathsat_smt_ast>(b)->a);
  check_msat_error(neg_b);

  return new_ast(
    msat_make_plus(env, to_solver_smt_ast<mathsat_smt_ast>(a)->a, neg_b),
    a->sort);
}

smt_astt mathsat_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_minus(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    msat_make_times(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_times(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_srem(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_urem(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_sdiv(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_udiv(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_lshl(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_ashr(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_lshr(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    msat_make_times(
      env,
      msat_make_number(env, "-1"),
      to_solver_smt_ast<mathsat_smt_ast>(a)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    msat_make_bv_neg(env, to_solver_smt_ast<mathsat_smt_ast>(a)->a), a->sort);
}

smt_astt mathsat_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    msat_make_bv_not(env, to_solver_smt_ast<mathsat_smt_ast>(a)->a), a->sort);
}

smt_astt mathsat_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_xor(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_or(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_and(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);

  // MathSAT doesn't seem to implement this; so do it manually. Following the
  // CNF conversion CBMC doet,shis is: lor(lnot(a), b)
  msat_term nota = msat_make_not(env, to_solver_smt_ast<mathsat_smt_ast>(a)->a);
  check_msat_error(nota);

  return new_ast(
    msat_make_or(env, nota, to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());

  // Another thing that mathsat doesn't implement.
  // Do this as and(or(a,b),not(and(a,b)))
  msat_term and2 = msat_make_and(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(a)->a,
    to_solver_smt_ast<mathsat_smt_ast>(b)->a);
  check_msat_error(and2);

  msat_term notand2 = msat_make_not(env, and2);
  check_msat_error(notand2);

  msat_term or1 = msat_make_or(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(a)->a,
    to_solver_smt_ast<mathsat_smt_ast>(b)->a);
  check_msat_error(or1);

  return new_ast(msat_make_and(env, or1, notand2), a->sort);
}

smt_astt mathsat_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    msat_make_or(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    msat_make_and(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    msat_make_not(env, to_solver_smt_ast<mathsat_smt_ast>(a)->a), boolean_sort);
}

smt_astt mathsat_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    msat_make_bv_slt(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_ult(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_slt(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    msat_make_leq(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_uleq(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_bv_sleq(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  if(a->sort->id == SMT_SORT_BOOL || a->sort->id == SMT_SORT_STRUCT)
    return new_ast(
      msat_make_iff(
        env,
        to_solver_smt_ast<mathsat_smt_ast>(a)->a,
        to_solver_smt_ast<mathsat_smt_ast>(b)->a),
      boolean_sort);

  return new_ast(
    msat_make_equal(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt mathsat_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());

  return new_ast(
    msat_make_array_write(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a,
      to_solver_smt_ast<mathsat_smt_ast>(c)->a),
    a->sort);
}

smt_astt mathsat_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    msat_make_array_read(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(a)->a,
      to_solver_smt_ast<mathsat_smt_ast>(b)->a),
    a->sort->get_range_sort());
}

smt_astt mathsat_convt::mk_smt_int(
  const mp_integer &theint,
  bool sign __attribute__((unused)))
{
  char buffer[256], *n = nullptr;
  n = theint.as_string(buffer, 256);
  assert(n != nullptr);

  msat_term t = msat_make_number(env, n);
  check_msat_error(t);

  smt_sortt s = mk_int_sort();
  return new_ast(t, s);
}

smt_ast *mathsat_convt::mk_smt_real(const std::string &str)
{
  msat_term t = msat_make_number(env, str.c_str());
  check_msat_error(t);

  smt_sortt s = mk_real_sort();
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_smt_bv(const mp_integer &theint, smt_sortt s)
{
  std::size_t w = s->get_data_width();

  // MathSAT refuses to parse negative integers. So, feed it binary.
  std::string str = integer2binary(theint, w);

  // Make bv int from textual representation.
  msat_term t = msat_make_bv_number(env, str.c_str(), w, 2);
  check_msat_error(t);

  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  msat_term t;
  if(use_fp_api)
  {
    std::size_t w = thereal.spec.e + thereal.spec.f + 1;
    std::string str = integer2binary(thereal.pack(), w);
    t = msat_make_bv_number(env, str.c_str(), w, 2);
  }
  else
  {
    const mp_integer sig = thereal.get_fraction();

    // If the number is denormal, we set the exponent to 0
    const mp_integer exp =
      thereal.is_normal() ? thereal.get_exponent() + thereal.spec.bias() : 0;

    std::string sgn_str = thereal.get_sign() ? "1" : "0";
    std::string exp_str = integer2binary(exp, thereal.spec.e);
    std::string sig_str = integer2binary(sig, thereal.spec.f);

    std::string smt_str = "(fp #b" + sgn_str;
    smt_str += " #b" + exp_str;
    smt_str += " #b" + sig_str;
    smt_str += ")";

    t = msat_from_string(env, smt_str.c_str());
  }
  check_msat_error(t);

  smt_sortt s = mk_real_fp_sort(thereal.spec.e, thereal.spec.f);
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_nan(ew, sw);

  smt_sortt s = mk_real_fp_sort(ew, sw - 1);

  msat_term t = msat_make_fp_nan(env, ew, sw - 1);
  check_msat_error(t);

  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_inf(sgn, ew, sw);

  smt_sortt s = mk_real_fp_sort(ew, sw - 1);

  msat_term t = sgn ? msat_make_fp_minus_inf(env, ew, sw - 1)
                    : msat_make_fp_plus_inf(env, ew, sw - 1);
  check_msat_error(t);

  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_rm(rm);

  msat_term t;
  switch(rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    t = msat_make_fp_roundingmode_nearest_even(env);
    break;
  case ieee_floatt::ROUND_TO_MINUS_INF:
    t = msat_make_fp_roundingmode_minus_inf(env);
    break;
  case ieee_floatt::ROUND_TO_PLUS_INF:
    t = msat_make_fp_roundingmode_plus_inf(env);
    break;
  case ieee_floatt::ROUND_TO_ZERO:
    t = msat_make_fp_roundingmode_zero(env);
    break;
  default:
    abort();
  }
  check_msat_error(t);

  return new_ast(t, mk_fpbv_rm_sort());
}

smt_ast *mathsat_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = boolean_sort;
  return new mathsat_smt_ast(
    this, s, (val) ? msat_make_true(env) : msat_make_false(env));
}

smt_ast *mathsat_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *mathsat_convt::mk_smt_symbol(
  const std::string &name,
  const smt_sort *s)
{
  // XXX - does 'd' leak?
  msat_decl d = msat_declare_function(
    env, name.c_str(), to_solver_smt_sort<msat_type>(s)->s);
  assert(!MSAT_ERROR_DECL(d) && "Invalid function symbol declaration sort");

  msat_term t = msat_make_constant(env, d);
  check_msat_error(t);
  return new_ast(t, s);
}

smt_astt
mathsat_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low)
{
  // If it's a floatbv, convert it to bv
  if(a->sort->id == SMT_SORT_FPBV)
    a = mk_from_fp_to_bv(a);

  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);

  msat_term t = msat_make_bv_extract(env, high, low, mast->a);
  check_msat_error(t);

  smt_sortt s = mk_bv_sort(high - low + 1);
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);

  msat_term t = msat_make_bv_sext(env, topwidth, mast->a);
  check_msat_error(t);

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);

  msat_term t = msat_make_bv_zext(env, topwidth, mast->a);
  check_msat_error(t);

  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_concat(smt_astt a, smt_astt b)
{
  msat_term t = msat_make_bv_concat(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(a)->a,
    to_solver_smt_ast<mathsat_smt_ast>(b)->a);
  check_msat_error(t);

  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());
  return new_ast(t, s);
}

smt_astt mathsat_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  msat_term r;
  if(t->sort->id == SMT_SORT_BOOL)
  {
    // MathSAT shows a dislike of implementing this with booleans. Follow
    // CBMC's CNF flattening and make this
    // (with c = cond, t = trueval, f = falseval):
    //
    //   or(and(c,t),and(not(c), f))
    msat_term land1 = msat_make_and(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(cond)->a,
      to_solver_smt_ast<mathsat_smt_ast>(t)->a);
    check_msat_error(land1);

    msat_term notval =
      msat_make_not(env, to_solver_smt_ast<mathsat_smt_ast>(cond)->a);
    check_msat_error(notval);

    msat_term land2 =
      msat_make_and(env, notval, to_solver_smt_ast<mathsat_smt_ast>(f)->a);
    check_msat_error(land2);

    r = msat_make_or(env, land1, land2);
  }
  else
  {
    r = msat_make_term_ite(
      env,
      to_solver_smt_ast<mathsat_smt_ast>(cond)->a,
      to_solver_smt_ast<mathsat_smt_ast>(t)->a,
      to_solver_smt_ast<mathsat_smt_ast>(f)->a);
  }

  check_msat_error(r);

  return new_ast(r, t->sort);
}

const smt_ast *
mathsat_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void mathsat_convt::add_array_constraints_for_solving()
{
}

void mathsat_convt::push_array_ctx()
{
}

void mathsat_convt::pop_array_ctx()
{
}

void mathsat_smt_ast::dump() const
{
  // We need to get the env
  auto convt = dynamic_cast<const mathsat_convt *>(context);
  assert(convt != nullptr);

  std::cout << msat_to_smtlib2(convt->env, a) << std::endl;
}

void mathsat_convt::dump_smt()
{
  size_t num_of_asserted;
  msat_term *asserted_formulas =
    msat_get_asserted_formulas(env, &num_of_asserted);

  for(unsigned i = 0; i < num_of_asserted; i++)
    std::cout << msat_to_smtlib2(env, asserted_formulas[i]) << "\n";

  msat_free(asserted_formulas);
}

smt_astt mathsat_convt::mk_smt_fpbv_fma(
  smt_astt v1,
  smt_astt v2,
  smt_astt v3,
  smt_astt rm)
{
  // MathSAT does not support FMA, so convert to BVFP and call the fp_api

  // Convert the rounding mode
  smt_astt is_ne = mk_eq(rm, mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_EVEN));
  smt_astt is_mi = mk_eq(rm, mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_MINUS_INF));
  smt_astt is_pi = mk_eq(rm, mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_PLUS_INF));

  bool old_use_fp_api = use_fp_api;
  use_fp_api = true;

  // We don't need to check if we're running in the fp2bv mode, as
  // mk_from_fp_to_bv doesn't do anything in that mode
  smt_astt new_v1 = mk_from_fp_to_bv(v1);
  smt_astt new_v2 = mk_from_fp_to_bv(v2);
  smt_astt new_v3 = mk_from_fp_to_bv(v3);

  smt_astt new_rm = mk_ite(
    is_ne,
    mk_smt_bv(BigInt(0), mk_bvfp_rm_sort()),
    mk_ite(
      is_pi,
      mk_smt_bv(BigInt(2), mk_bvfp_rm_sort()),
      mk_ite(
        is_mi,
        mk_smt_bv(BigInt(3), mk_bvfp_rm_sort()),
        mk_smt_bv(BigInt(4), mk_bvfp_rm_sort()))));

  // Call fma
  smt_astt fma = fp_convt::mk_smt_fpbv_fma(new_v1, new_v2, new_v3, new_rm);

  use_fp_api = old_use_fp_api;

  // And convert back to FPBV. Again, no need to check if we're running in
  // fp2bv mode, as mk_from_bv_to_fp doesn't do anything in fp2bv mode
  return mk_from_bv_to_fp(fma, v1->sort);
}

void mathsat_convt::print_model()
{
  /* we use a model iterator to retrieve the model values for all the
     * variables, and the necessary function instantiations */
  msat_model_iterator iter = msat_create_model_iterator(env);
  assert(!MSAT_ERROR_MODEL_ITERATOR(iter));

  printf("Model:\n");
  while(msat_model_iterator_has_next(iter))
  {
    msat_term t, v;
    char *s;
    msat_model_iterator_next(iter, &t, &v);
    s = msat_term_repr(t);
    assert(s);
    printf(" %s = ", s);
    msat_free(s);
    s = msat_term_repr(v);
    assert(s);
    printf("%s\n", s);
    msat_free(s);
  }
  msat_destroy_model_iterator(iter);
}

smt_sortt mathsat_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  if(use_fp_api)
    return fp_convt::mk_fpbv_sort(ew, sw);

  auto t = msat_get_fp_type(env, ew, sw);
  return new solver_smt_sort<msat_type>(SMT_SORT_FPBV, t, ew + sw + 1, sw + 1);
}

smt_sortt mathsat_convt::mk_fpbv_rm_sort()
{
  if(use_fp_api)
    return mk_bvfp_rm_sort();

  auto t = msat_get_fp_roundingmode_type(env);
  return new solver_smt_sort<msat_type>(SMT_SORT_FPBV_RM, t, 3);
}

smt_sortt mathsat_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_BVFP, msat_get_bv_type(env, ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt mathsat_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_BVFP_RM, msat_get_bv_type(env, 3), 3);
}

smt_sortt mathsat_convt::mk_bool_sort()
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_BOOL, msat_get_bool_type(env), 1);
}

smt_sortt mathsat_convt::mk_real_sort()
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_REAL, msat_get_rational_type(env), 0);
}

smt_sortt mathsat_convt::mk_int_sort()
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_INT, msat_get_integer_type(env), 0);
}

smt_sortt mathsat_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_BV, msat_get_bv_type(env, width), width);
}

smt_sortt mathsat_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<msat_type>(
    SMT_SORT_FIXEDBV, msat_get_bv_type(env, width), width);
}

smt_sortt mathsat_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<msat_type>(domain);
  auto range_sort = to_solver_smt_sort<msat_type>(range);

  auto t = msat_get_array_type(env, domain_sort->s, range_sort->s);
  return new solver_smt_sort<msat_type>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_astt mathsat_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  if(use_fp_api)
    return fp_convt::mk_from_bv_to_fp(op, to);

  msat_term t = msat_make_fp_from_ieeebv(
    env,
    to->get_exponent_width(),
    to->get_significand_width() - 1,
    to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(t);

  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_from_fp_to_bv(smt_astt op)
{
  msat_term t =
    msat_make_fp_as_ieeebv(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(t);

  smt_sortt to = mk_bvfp_sort(
    op->sort->get_exponent_width(), op->sort->get_significand_width() - 1);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_ubv(
  smt_astt from,
  std::size_t width)
{
  if(use_fp_api)
    return fp_convt::mk_smt_typecast_from_fpbv_to_ubv(from, width);

  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  msat_term t = msat_make_fp_to_bv(
    env,
    width,
    to_solver_smt_ast<mathsat_smt_ast>(
      mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO))
      ->a,
    mfrom->a);
  check_msat_error(t);

  smt_sortt to = mk_bv_sort(width);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_sbv(
  smt_astt from,
  std::size_t width)
{
  if(use_fp_api)
    return fp_convt::mk_smt_typecast_from_fpbv_to_sbv(from, width);

  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  msat_term t = msat_make_fp_to_bv(
    env,
    width,
    to_solver_smt_ast<mathsat_smt_ast>(
      mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO))
      ->a,
    mfrom->a);
  check_msat_error(t);

  smt_sortt to = mk_bv_sort(width);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_typecast_from_fpbv_to_fpbv(from, to, rm);

  unsigned sw = to->get_significand_width() - 1;
  unsigned ew = to->get_exponent_width();

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_cast(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_typecast_ubv_to_fpbv(from, to, rm);

  unsigned sw = to->get_significand_width() - 1;
  unsigned ew = to->get_exponent_width();

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_from_ubv(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_typecast_sbv_to_fpbv(from, to, rm);

  unsigned sw = to->get_significand_width() - 1;
  unsigned ew = to->get_exponent_width();

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_from_sbv(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new_ast(t, to);
}

smt_astt mathsat_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_nearbyint_from_float(from, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  // Conversion from float, using the correct rounding mode
  msat_term t = msat_make_fp_round_to_int(env, mrm->a, mfrom->a);
  return new_ast(t, from->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_sqrt(rd, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mrd = to_solver_smt_ast<mathsat_smt_ast>(rd);

  msat_term t = msat_make_fp_sqrt(env, mrm->a, mrd->a);
  check_msat_error(t);

  return new_ast(t, rd->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_add(lhs, rhs, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_plus(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new_ast(t, lhs->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_sub(lhs, rhs, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_minus(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new_ast(t, lhs->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_mul(lhs, rhs, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_times(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new_ast(t, lhs->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_div(lhs, rhs, rm);

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_div(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new_ast(t, lhs->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_eq(lhs, rhs);

  msat_term r = msat_make_fp_equal(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(lhs)->a,
    to_solver_smt_ast<mathsat_smt_ast>(rhs)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_lt(lhs, rhs);

  msat_term r = msat_make_fp_lt(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(lhs)->a,
    to_solver_smt_ast<mathsat_smt_ast>(rhs)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_lte(lhs, rhs);

  msat_term r = msat_make_fp_leq(
    env,
    to_solver_smt_ast<mathsat_smt_ast>(lhs)->a,
    to_solver_smt_ast<mathsat_smt_ast>(rhs)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_neg(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_neg(op);

  msat_term r =
    msat_make_fp_neg(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, op->sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_nan(op);

  msat_term r =
    msat_make_fp_isnan(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_inf(op);

  msat_term r =
    msat_make_fp_isinf(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_normal(op);

  msat_term r =
    msat_make_fp_isnormal(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_zero(op);

  msat_term r =
    msat_make_fp_iszero(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_negative(op);

  msat_term r =
    msat_make_fp_isneg(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_is_positive(op);

  msat_term r =
    msat_make_fp_ispos(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, boolean_sort);
}

smt_astt mathsat_convt::mk_smt_fpbv_abs(smt_astt op)
{
  if(use_fp_api)
    return fp_convt::mk_smt_fpbv_abs(op);

  msat_term r =
    msat_make_fp_abs(env, to_solver_smt_ast<mathsat_smt_ast>(op)->a);
  check_msat_error(r);

  return new_ast(r, op->sort);
}
