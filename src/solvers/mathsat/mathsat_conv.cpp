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
  : smt_convt(int_encoding, ns), array_iface(false, false), fp_convt(this)
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

expr2tc mathsat_convt::get_bool(const smt_ast *a)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);
  msat_term t = msat_get_model_value(env, mast->a);
  check_msat_error(t);

  expr2tc res;
  if(msat_term_is_true(env, t))
    res = gen_true_expr();
  else if(msat_term_is_false(env, t))
    res = gen_false_expr();
  else
  {
    std::cerr << "Boolean model value is neither true or false" << std::endl;
    abort();
  }

  msat_free(msat_term_repr(t));
  return res;
}

expr2tc mathsat_convt::get_bv(const type2tc &type, smt_astt a)
{
  assert(a->sort->id >= SMT_SORT_SBV || a->sort->id <= SMT_SORT_FIXEDBV);

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

  return build_bv(type, BigInt(finval));
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
  int ret =
    msat_is_fp_type(env, to_solver_smt_sort<msat_type>(a->sort)->s, &ew, &sw);
  assert(ret != 0 && "Non FP type passed to mathsat_convt::get_exp_width");

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

  smt_astt tmpast = mk_smt_bvint(BigInt(index), false, orig_w);
  const mathsat_smt_ast *tmpa = to_solver_smt_ast<mathsat_smt_ast>(tmpast);

  msat_term t = msat_make_array_read(env, mast->a, tmpa->a);
  check_msat_error(t);

  mathsat_smt_ast *tmpb = new mathsat_smt_ast(this, convert_sort(subtype), t);
  expr2tc result = get_bv(subtype, tmpb);
  free(tmpb);

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

smt_ast *mathsat_convt::mk_func_app(
  const smt_sort *s,
  smt_func_kind k,
  const smt_ast *const *_args,
  unsigned int numargs)
{
  const mathsat_smt_ast *args[4];
  unsigned int i;

  assert(numargs <= 4);
  for(i = 0; i < numargs; i++)
    args[i] = to_solver_smt_ast<mathsat_smt_ast>(_args[i]);

  msat_term r;

  switch(k)
  {
  case SMT_FUNC_EQ:
    // MathSAT demands we use iff for boolean equivalence.
    if(args[0]->sort->id == SMT_SORT_BOOL)
      r = msat_make_iff(env, args[0]->a, args[1]->a);
    else
      r = msat_make_equal(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_NOTEQ:
  {
    smt_ast *a = mk_func_app(s, SMT_FUNC_EQ, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_NOT:
    r = msat_make_not(env, args[0]->a);
    break;
  case SMT_FUNC_AND:
    r = msat_make_and(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_OR:
    r = msat_make_or(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_XOR:
  {
    // Another thing that mathsat doesn't implement.
    // Do this as and(or(a,b),not(and(a,b)))
    msat_term and2 = msat_make_and(env, args[0]->a, args[1]->a);
    check_msat_error(and2);

    msat_term notand2 = msat_make_not(env, and2);
    check_msat_error(notand2);

    msat_term or1 = msat_make_or(env, args[0]->a, args[1]->a);
    check_msat_error(or1);

    r = msat_make_and(env, or1, notand2);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    // MathSAT doesn't seem to implement this; so do it manually. Following the
    // CNF conversion CBMC does, this is: lor(lnot(a), b)
    msat_term nota = msat_make_not(env, args[0]->a);
    check_msat_error(nota);

    r = msat_make_or(env, nota, args[1]->a);
    break;
  }
  case SMT_FUNC_ITE:
    if(s->id == SMT_SORT_BOOL)
    {
      // MathSAT shows a dislike of implementing this with booleans. Follow
      // CBMC's CNF flattening and make this
      // (with c = cond, t = trueval, f = falseval):
      //
      //   or(and(c,t),and(not(c), f))
      msat_term land1 = msat_make_and(env, args[0]->a, args[1]->a);
      check_msat_error(land1);

      msat_term notval = msat_make_not(env, args[0]->a);
      check_msat_error(notval);

      msat_term land2 = msat_make_and(env, notval, args[2]->a);
      check_msat_error(land2);

      r = msat_make_or(env, land1, land2);
    }
    else
    {
      r = msat_make_term_ite(env, args[0]->a, args[1]->a, args[2]->a);
    }
    break;
  case SMT_FUNC_CONCAT:
    r = msat_make_bv_concat(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVNOT:
    r = msat_make_bv_not(env, args[0]->a);
    break;
  case SMT_FUNC_NEG:
  case SMT_FUNC_BVNEG:
    if(s->id == SMT_SORT_FLOATBV)
    {
      r = msat_make_fp_neg(env, args[0]->a);
    }
    else
    {
      r = msat_make_bv_neg(env, args[0]->a);
    }
    break;
  case SMT_FUNC_BVAND:
    r = msat_make_bv_and(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVOR:
    r = msat_make_bv_or(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVXOR:
    r = msat_make_bv_xor(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_ADD:
    r = msat_make_plus(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_SUB:
  {
    msat_term neg_b =
      msat_make_times(env, msat_make_number(env, "-1"), args[1]->a);
    check_msat_error(neg_b);

    r = msat_make_plus(env, args[0]->a, neg_b);
    break;
  }
  case SMT_FUNC_BVADD:
    r = msat_make_bv_plus(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSUB:
    r = msat_make_bv_minus(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_MUL:
    r = msat_make_times(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVMUL:
    r = msat_make_bv_times(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSDIV:
    r = msat_make_bv_sdiv(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUDIV:
    r = msat_make_bv_udiv(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSMOD:
    r = msat_make_bv_srem(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUMOD:
    r = msat_make_bv_urem(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSHL:
    r = msat_make_bv_lshl(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVLSHR:
    r = msat_make_bv_lshr(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVASHR:
    r = msat_make_bv_ashr(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUGTE:
  {
    // This is !ULT
    assert(s->id == SMT_SORT_BOOL);
    const smt_ast *a = mk_func_app(s, SMT_FUNC_BVULT, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_BVUGT:
  {
    // This is !ULTE
    assert(s->id == SMT_SORT_BOOL);
    const smt_ast *a = mk_func_app(s, SMT_FUNC_BVULTE, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_BVULTE:
    r = msat_make_bv_uleq(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVULT:
    r = msat_make_bv_ult(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVSGTE:
  {
    // This is !SLT
    assert(s->id == SMT_SORT_BOOL);
    const smt_ast *a = mk_func_app(s, SMT_FUNC_BVSLT, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_GT:
  case SMT_FUNC_BVSGT:
  {
    assert(s->id == SMT_SORT_BOOL);

    // (a > b) iff (b < a)
    if(
      (args[0]->sort->id == SMT_SORT_FLOATBV) &&
      (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_lt(env, args[1]->a, args[0]->a);
    else
      r = msat_make_bv_slt(env, args[1]->a, args[0]->a);
    break;
  }
  case SMT_FUNC_LTE:
    if(
      (args[0]->sort->id == SMT_SORT_FLOATBV) &&
      (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_leq(env, args[0]->a, args[1]->a);
    else
      r = msat_make_leq(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSLTE:
    r = msat_make_bv_sleq(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_LT:
  case SMT_FUNC_BVSLT:
    if(
      (args[0]->sort->id == SMT_SORT_FLOATBV) &&
      (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_lt(env, args[0]->a, args[1]->a);
    else
      r = msat_make_bv_slt(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_STORE:
    r = msat_make_array_write(env, args[0]->a, args[1]->a, args[2]->a);
    break;
  case SMT_FUNC_SELECT:
    r = msat_make_array_read(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_ISZERO:
    r = msat_make_fp_iszero(env, args[0]->a);
    break;
  case SMT_FUNC_FABS:
    r = msat_make_fp_abs(env, args[0]->a);
    break;
  case SMT_FUNC_ISNAN:
    r = msat_make_fp_isnan(env, args[0]->a);
    break;
  case SMT_FUNC_ISINF:
    r = msat_make_fp_isinf(env, args[0]->a);
    break;
  case SMT_FUNC_ISNORMAL:
    r = msat_make_fp_isnormal(env, args[0]->a);
    break;
  case SMT_FUNC_ISNEG:
    r = msat_make_fp_isneg(env, args[0]->a);
    break;
  case SMT_FUNC_ISPOS:
    r = msat_make_fp_ispos(env, args[0]->a);
    break;
  case SMT_FUNC_IEEE_EQ:
    r = msat_make_fp_equal(env, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BV2FLOAT:
  {
    unsigned sw = s->get_significand_width();
    unsigned ew = s->get_data_width() - sw;
    r = msat_make_fp_from_ieeebv(env, ew, sw, args[0]->a);
    break;
  }
  case SMT_FUNC_FLOAT2BV:
    r = msat_make_fp_as_ieeebv(env, args[0]->a);
    break;
  default:
    std::cerr << "Unhandled SMT function \"" << smt_func_name_table[k] << "\" "
              << "in mathsat conversion" << std::endl;
    abort();
  }
  check_msat_error(r);

  return new mathsat_smt_ast(this, s, r);
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
  return new mathsat_smt_ast(this, s, t);
}

smt_ast *mathsat_convt::mk_smt_real(const std::string &str)
{
  msat_term t = msat_make_number(env, str.c_str());
  check_msat_error(t);

  smt_sortt s = mk_real_sort();
  return new mathsat_smt_ast(this, s, t);
}

smt_astt
mathsat_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w)
{
  std::stringstream ss;

  // MathSAT refuses to parse negative integers. So, feed it binary.
  std::string str = integer2binary(theint, w);

  // Make bv int from textual representation.
  msat_term t = msat_make_bv_number(env, str.c_str(), w, 2);
  check_msat_error(t);

  smt_sortt s = mk_int_bv_sort(sign ? SMT_SORT_SBV : SMT_SORT_UBV, w);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_fpbv(const ieee_floatt &thereal)
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

  msat_term t = msat_from_string(env, smt_str.c_str());
  check_msat_error(t);

  smt_sortt s = mk_real_fp_sort(thereal.spec.e, thereal.spec.f);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw);
  unsigned swidth = s->get_significand_width();
  unsigned ewidth = s->get_data_width() - swidth;

  msat_term t = msat_make_fp_nan(env, ewidth, swidth);
  check_msat_error(t);

  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw);
  unsigned swidth = s->get_significand_width();
  unsigned ewidth = s->get_data_width() - swidth;

  msat_term t = sgn ? msat_make_fp_minus_inf(env, ewidth, swidth)
                    : msat_make_fp_plus_inf(env, ewidth, swidth);
  check_msat_error(t);

  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
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

  return new mathsat_smt_ast(this, mk_fpbv_rm_sort(), t);
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
  return new mathsat_smt_ast(this, s, t);
}

smt_ast *mathsat_convt::mk_extract(
  const smt_ast *a,
  unsigned int high,
  unsigned int low,
  const smt_sort *s)
{
  const mathsat_smt_ast *mast = to_solver_smt_ast<mathsat_smt_ast>(a);

  // If it's a floatbv, convert it to bv
  if(a->sort->id == SMT_SORT_FLOATBV)
  {
    msat_term t = msat_make_fp_as_ieeebv(env, mast->a);
    check_msat_error(t);

    smt_ast *bv = new mathsat_smt_ast(this, s, t);
    mast = to_solver_smt_ast<mathsat_smt_ast>(bv);
  }

  msat_term t = msat_make_bv_extract(env, high, low, mast->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, s, t);
}

const smt_ast *mathsat_convt::convert_array_of(
  smt_astt init_val,
  unsigned long domain_width)
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
  auto t = msat_get_fp_type(env, ew, sw);
  return new solver_smt_sort<msat_type>(SMT_SORT_FLOATBV, t, ew + sw, sw);
}

smt_sortt mathsat_convt::mk_fpbv_rm_sort()
{
  auto t = msat_get_fp_roundingmode_type(env);
  return new solver_smt_sort<msat_type>(SMT_SORT_FLOATBV_RM, t, 1);
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

smt_sortt mathsat_convt::mk_bv_sort(const smt_sort_kind k, std::size_t width)
{
  return new solver_smt_sort<msat_type>(k, msat_get_bv_type(env, width), width);
}

smt_sortt mathsat_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<msat_type>(domain);
  auto range_sort = to_solver_smt_sort<msat_type>(range);

  auto t = msat_get_array_type(env, domain_sort->s, range_sort->s);
  return new solver_smt_sort<msat_type>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_ubv(
  smt_astt from,
  smt_sortt to)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(
    mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_to_bv(env, to->get_data_width(), mrm->a, mfrom->a);
  check_msat_error(t);
  return new mathsat_smt_ast(this, to, t);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_sbv(
  smt_astt from,
  smt_sortt to)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(
    mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_to_bv(env, to->get_data_width(), mrm->a, mfrom->a);
  check_msat_error(t);
  return new mathsat_smt_ast(this, to, t);
}

smt_astt mathsat_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_data_width() - sw;

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_cast(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new mathsat_smt_ast(this, to, t);
}

smt_astt mathsat_convt::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_data_width() - sw;

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_from_ubv(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new mathsat_smt_ast(this, to, t);
}

smt_astt mathsat_convt::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  unsigned sw = to->get_significand_width();
  unsigned ew = to->get_data_width() - sw;

  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  msat_term t = msat_make_fp_from_sbv(env, ew, sw, mrm->a, mfrom->a);
  check_msat_error(t);
  return new mathsat_smt_ast(this, to, t);
}

smt_astt mathsat_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mfrom = to_solver_smt_ast<mathsat_smt_ast>(from);

  // Conversion from float, using the correct rounding mode
  msat_term t = msat_make_fp_round_to_int(env, mrm->a, mfrom->a);
  return new mathsat_smt_ast(this, from->sort, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mrd = to_solver_smt_ast<mathsat_smt_ast>(rd);

  msat_term t = msat_make_fp_sqrt(env, mrm->a, mrd->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, rd->sort, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_plus(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, lhs->sort, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_minus(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, lhs->sort, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_times(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, lhs->sort, t);
}

smt_astt mathsat_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const mathsat_smt_ast *mrm = to_solver_smt_ast<mathsat_smt_ast>(rm);
  const mathsat_smt_ast *mlhs = to_solver_smt_ast<mathsat_smt_ast>(lhs);
  const mathsat_smt_ast *mrhs = to_solver_smt_ast<mathsat_smt_ast>(rhs);

  msat_term t = msat_make_fp_div(env, mrm->a, mlhs->a, mrhs->a);
  check_msat_error(t);

  return new mathsat_smt_ast(this, lhs->sort, t);
}
