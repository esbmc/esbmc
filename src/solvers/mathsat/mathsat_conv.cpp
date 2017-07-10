#include <gmp.h>
#include <mathsat_conv.h>
#include <sstream>
#include <string>
#include <util/c_types.h>
#include <util/expr_util.h>

static const char* mathsat_config =
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
  "theory.arr.permanent_lemma_inst = true\n"
  "theory.arr.enable_witness = true";

// Ahem
msat_env* _env = nullptr;

void print_mathsat_formula()
{
  size_t num_of_asserted;
  msat_term * asserted_formulas =
   msat_get_asserted_formulas(*_env, &num_of_asserted);

  for (unsigned i=0; i< num_of_asserted; i++)
    std::cout << msat_to_smtlib2(*_env, asserted_formulas[i]) << "\n";

  msat_free(asserted_formulas);
}

void check_msat_error(msat_term &r)
{
  if (MSAT_ERROR_TERM(r)) {
    std::cerr << "Error creating SMT " << std::endl;
    std::cerr << "Error text: \"" << msat_last_error_message(*_env) << "\""
              << std::endl;
    abort();
  }
}

smt_convt *
create_new_mathsat_solver(bool int_encoding, const namespacet &ns,
                          const optionst &opts __attribute__((unused)),
                          tuple_iface **tuple_api __attribute__((unused)),
                          array_iface **array_api)
{
  mathsat_convt *conv = new mathsat_convt(int_encoding, ns);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

mathsat_convt::mathsat_convt(bool int_encoding,
                             const namespacet &ns)
  : smt_convt(int_encoding, ns), array_iface(false, false)
{
  cfg = msat_parse_config(mathsat_config);
  msat_set_option(cfg, "model_generation", "true");
  env = msat_create_env(cfg);
  _env = &env;
}

mathsat_convt::~mathsat_convt()
{
  msat_destroy_env(env);
  _env = nullptr;
}

void
mathsat_convt::push_ctx()
{
  smt_convt::push_ctx();
  msat_push_backtrack_point(env);
}

void
mathsat_convt::pop_ctx()
{
  msat_pop_backtrack_point(env);
  smt_convt::pop_ctx();
}

void
mathsat_convt::assert_ast(const smt_ast *a)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  msat_assert_formula(env, mast->t);
}

smt_convt::resultt
mathsat_convt::dec_solve()
{
  pre_solve();

  msat_result r = msat_solve(env);
  if (r == MSAT_SAT) {
    return P_SATISFIABLE;
  } else if (r == MSAT_UNSAT) {
    return P_UNSATISFIABLE;
  } else {
    std::cerr << "MathSAT returned MSAT_UNKNOWN for formula" << std::endl;
    abort();
  }
}

expr2tc
mathsat_convt::get_bool(const smt_ast *a)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  msat_term t = msat_get_model_value(env, mast->t);
  check_msat_error(t);

  if (msat_term_is_true(env, t)) {
    return gen_true_expr();
  } else if (msat_term_is_false(env, t)) {
    return gen_false_expr();
  } else {
    std::cerr << "Boolean model value is neither true or false" << std::endl;
    abort();
  }

  msat_free(msat_term_repr(t));
}

expr2tc
mathsat_convt::get_bv(const type2tc &_t,
                      const smt_ast *a)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  msat_term t = msat_get_model_value(env, mast->t);
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

  if(is_floatbv_type(_t))
  {
    ieee_float_spect spec(
      to_floatbv_type(_t).fraction,
      to_floatbv_type(_t).exponent);

    ieee_floatt number(spec);
    number.unpack(BigInt(buffer));

    return constant_floatbv2tc(number);
  }

  char *foo = buffer;
  int64_t finval = strtoll(buffer, &foo, 10);

  if (buffer[0] != '\0' && (foo == buffer || *foo != '\0')) {
    std::cerr << "Couldn't parse string representation of number \""
              << buffer << "\"" << std::endl;
    abort();
  }

  return constant_int2tc(get_uint64_type(), BigInt(finval));
}

expr2tc
mathsat_convt::get_array_elem(const smt_ast *array, uint64_t idx,
                              const type2tc &elem_sort)
{
  size_t orig_w = array->sort->domain_width;
  const mathsat_smt_ast *mast = mathsat_ast_downcast(array);

  smt_ast *tmpast = mk_smt_bvint(BigInt(idx), false, orig_w);
  const mathsat_smt_ast *tmpa = mathsat_ast_downcast(tmpast);

  msat_term t = msat_make_array_read(env, mast->t, tmpa->t);
  check_msat_error(t);
  free(tmpast);

  mathsat_smt_ast *tmpb = new mathsat_smt_ast(this, convert_sort(elem_sort), t);
  expr2tc result = get_bv(elem_sort, tmpb);
  free(tmpb);

  msat_free(msat_term_repr(t));

  return result;
}

tvt
mathsat_convt::l_get(const smt_ast *a)
{
  constant_bool2tc b = get_bool(a);
  if (b->value)
    return tvt(true);
  else if (!b->value)
    return tvt(false);
  else
    assert(0);
}

const std::string
mathsat_convt::solver_text()
{
  std::stringstream ss;
  char * tmp = msat_get_version();
  ss << tmp;
  msat_free(tmp);
  return ss.str();
}

smt_ast *
mathsat_convt::mk_func_app(const smt_sort *s, smt_func_kind k,
                             const smt_ast * const *_args,
                             unsigned int numargs)
{
  const mathsat_smt_ast *args[4];
  unsigned int i;

  assert(numargs <= 4);
  for (i = 0; i < numargs; i++)
    args[i] = mathsat_ast_downcast(_args[i]);

  msat_term r;

  switch (k) {
  case SMT_FUNC_EQ:
    // MathSAT demands we use iff for boolean equivalence.
    if (args[0]->sort->id == SMT_SORT_BOOL)
      r = msat_make_iff(env, args[0]->t, args[1]->t);
    else
      r = msat_make_equal(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_NOTEQ:
  {
    smt_ast *a = mk_func_app(s, SMT_FUNC_EQ, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_NOT:
    r = msat_make_not(env, args[0]->t);
    break;
  case SMT_FUNC_AND:
    r = msat_make_and(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_OR:
    r = msat_make_or(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_XOR:
  {
    // Another thing that mathsat doesn't implement.
    // Do this as and(or(a,b),not(and(a,b)))
    msat_term and2 = msat_make_and(env, args[0]->t, args[1]->t);
    check_msat_error(and2);

    msat_term notand2 = msat_make_not(env, and2);
    check_msat_error(notand2);

    msat_term or1 = msat_make_or(env, args[0]->t, args[1]->t);
    check_msat_error(or1);

    r = msat_make_and(env, or1, notand2);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    // MathSAT doesn't seem to implement this; so do it manually. Following the
    // CNF conversion CBMC does, this is: lor(lnot(a), b)
    msat_term nota = msat_make_not(env, args[0]->t);
    check_msat_error(nota);

    r = msat_make_or(env, nota, args[1]->t);
    break;
  }
  case SMT_FUNC_ITE:
    if (s->id == SMT_SORT_BOOL) {
      // MathSAT shows a dislike of implementing this with booleans. Follow
      // CBMC's CNF flattening and make this
      // (with c = cond, t = trueval, f = falseval):
      //
      //   or(and(c,t),and(not(c), f))
      msat_term land1 = msat_make_and(env, args[0]->t, args[1]->t);
      check_msat_error(land1);

      msat_term notval = msat_make_not(env, args[0]->t);
      check_msat_error(notval);

      msat_term land2 = msat_make_and(env, notval, args[2]->t);
      check_msat_error(land2);

      r = msat_make_or(env, land1, land2);
    } else {
      r = msat_make_term_ite(env, args[0]->t, args[1]->t, args[2]->t);
    }
    break;
  case SMT_FUNC_CONCAT:
    r = msat_make_bv_concat(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVNOT:
    r = msat_make_bv_not(env, args[0]->t);
    break;
  case SMT_FUNC_NEG:
  case SMT_FUNC_BVNEG:
    if (s->id == SMT_SORT_FLOATBV) {
      r = msat_make_fp_neg(env, args[0]->t);
    } else {
      r = msat_make_bv_neg(env, args[0]->t);
    }
    break;
  case SMT_FUNC_BVAND:
    r = msat_make_bv_and(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVOR:
    r = msat_make_bv_or(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVXOR:
    r = msat_make_bv_xor(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_ADD:
    r = msat_make_plus(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_SUB:
  {
    msat_term neg_b =
      msat_make_times(env, msat_make_number(env, "-1"), args[1]->t);
    check_msat_error(neg_b);

    r = msat_make_plus(env, args[0]->t, neg_b);
    break;
  }
  case SMT_FUNC_BVADD:
    r = msat_make_bv_plus(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSUB:
    r = msat_make_bv_minus(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_MUL:
    r = msat_make_times(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVMUL:
    r = msat_make_bv_times(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSDIV:
    r = msat_make_bv_sdiv(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVUDIV:
    r = msat_make_bv_udiv(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSMOD:
    r = msat_make_bv_srem(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVUMOD:
    r = msat_make_bv_urem(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSHL:
    r = msat_make_bv_lshl(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVLSHR:
    r = msat_make_bv_lshr(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVASHR:
    r = msat_make_bv_ashr(env, args[0]->t, args[1]->t);
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
    r = msat_make_bv_uleq(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVULT:
    r = msat_make_bv_ult(env, args[0]->t, args[1]->t);
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
    if((args[0]->sort->id == SMT_SORT_FLOATBV)
        && (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_lt(env, args[1]->t, args[0]->t);
    else
      r = msat_make_bv_slt(env, args[1]->t, args[0]->t);
    break;
  }
  case SMT_FUNC_LTE:
    if((args[0]->sort->id == SMT_SORT_FLOATBV)
        && (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_leq(env, args[0]->t, args[1]->t);
    else
      r = msat_make_leq(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSLTE:
      r = msat_make_bv_sleq(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_LT:
  case SMT_FUNC_BVSLT:
    if((args[0]->sort->id == SMT_SORT_FLOATBV)
        && (args[1]->sort->id == SMT_SORT_FLOATBV))
      r = msat_make_fp_lt(env, args[0]->t, args[1]->t);
    else
      r = msat_make_bv_slt(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_STORE:
    r = msat_make_array_write(env, args[0]->t, args[1]->t, args[2]->t);
    break;
  case SMT_FUNC_SELECT:
    r = msat_make_array_read(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_ISZERO:
    r = msat_make_fp_iszero(env, args[0]->t);
    break;
  case SMT_FUNC_FABS:
    r = msat_make_fp_abs(env, args[0]->t);
    break;
  case SMT_FUNC_ISNAN:
    r = msat_make_fp_isnan(env, args[0]->t);
    break;
  case SMT_FUNC_ISINF:
    r = msat_make_fp_isinf(env, args[0]->t);
    break;
  case SMT_FUNC_ISNORMAL:
    r = msat_make_fp_isnormal(env, args[0]->t);
    break;
  case SMT_FUNC_ISNEG:
    r = msat_make_fp_isneg(env, args[0]->t);
    break;
  case SMT_FUNC_ISPOS:
    r = msat_make_fp_ispos(env, args[0]->t);
    break;
  case SMT_FUNC_IEEE_EQ:
    r = msat_make_fp_equal(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BV2FLOAT:
    r = msat_make_fp_from_ieeebv(env, get_exp_width(s), get_mant_width(s), args[0]->t);
    break;
  case SMT_FUNC_FLOAT2BV:
    r = msat_make_fp_as_ieeebv(env, args[0]->t);
    break;
  default:
    std::cerr << "Unhandled SMT function \"" << smt_func_name_table[k] << "\" "
              << "in mathsat conversion" << std::endl;
    abort();
  }
  check_msat_error(r);

  return new mathsat_smt_ast(this, s, r);
}

smt_sort *
mathsat_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    return new mathsat_smt_sort(k, msat_get_integer_type(env));
  case SMT_SORT_REAL:
    return new mathsat_smt_sort(k, msat_get_rational_type(env));
  case SMT_SORT_BV:
  {
    unsigned long uint = va_arg(ap, unsigned long);
    return new mathsat_smt_sort(k, msat_get_bv_type(env, uint), uint);
  }
  case SMT_SORT_FLOATBV:
  {
    unsigned ew = va_arg(ap, unsigned long);
    unsigned sw = va_arg(ap, unsigned long);
    return new mathsat_smt_sort(k, msat_get_fp_type(env, ew, sw), ew + sw + 1);
  }
  case SMT_SORT_FLOATBV_RM:
    return new mathsat_smt_sort(k, msat_get_fp_roundingmode_type(env));
  case SMT_SORT_ARRAY:
  {
    const mathsat_smt_sort *dom = va_arg(ap, const mathsat_smt_sort *);
    const mathsat_smt_sort *range = va_arg(ap, const mathsat_smt_sort *);
    assert(int_encoding || dom->data_width != 0);

    // The range data width is allowed to be zero, which happens if the range
    // is not a bitvector / integer
    unsigned int data_width = range->data_width;
    if (range->id == SMT_SORT_STRUCT || range->id == SMT_SORT_BOOL || range->id == SMT_SORT_UNION)
      data_width = 1;

    return new mathsat_smt_sort(k, msat_get_array_type(env, dom->t, range->t),
                                data_width, dom->data_width, range);
  }
  case SMT_SORT_BOOL:
    return new mathsat_smt_sort(k, msat_get_bool_type(env));
  case SMT_SORT_STRUCT:
  case SMT_SORT_UNION:
    std::cerr << "MathSAT does not support tuples" << std::endl;
    abort();
  }

  std::cerr << "MathSAT sort conversion fell through" << std::endl;
  abort();
}

smt_ast *
mathsat_convt::mk_smt_int(const mp_integer &theint, bool sign __attribute__((unused)))
{
  char buffer[256], *n = nullptr;
  n = theint.as_string(buffer, 256);
  assert(n != nullptr);

  msat_term t = msat_make_number(env, n);
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_INT);
  return new mathsat_smt_ast(this, s, t);
}

smt_ast *
mathsat_convt::mk_smt_real(const std::string &str)
{
  msat_term t = msat_make_number(env, str.c_str());
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_REAL);
  return new mathsat_smt_ast(this, s, t);
}

smt_ast *
mathsat_convt::mk_smt_bvint(
  const mp_integer &theint,
  bool sign __attribute__((unused)),
  unsigned int w)
{
  std::stringstream ss;

  // MathSAT refuses to parse negative integers. So, feed it binary.
  std::string str = integer2binary(theint, w);

  // Make bv int from textual representation.
  msat_term t = msat_make_bv_number(env, str.c_str(), w, 2);
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_BV, w, false);
  return new mathsat_smt_ast(this, s, t);
}

smt_ast* mathsat_convt::mk_smt_bvfloat(const ieee_floatt &thereal,
                                       unsigned ew, unsigned sw)
{
  const mp_integer sig = thereal.get_fraction();

  // If the number is denormal, we set the exponent to 0
  const mp_integer exp = thereal.is_normal() ?
    thereal.get_exponent() + thereal.spec.bias() : 0;

  std::string sgn_str = thereal.get_sign() ? "1" : "0";
  std::string exp_str = integer2binary(exp, ew);
  std::string sig_str = integer2binary(sig, sw);

  std::string smt_str = "(fp #b" + sgn_str;
  smt_str += " #b" + exp_str;
  smt_str += " #b" + sig_str;
  smt_str += ")";

  msat_term t = msat_from_string(env, smt_str.c_str());
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_bvfloat_nan(unsigned ew, unsigned sw)
{
  msat_term t = msat_make_fp_nan(env, ew, sw);
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw)
{
  msat_term t =
    sgn ? msat_make_fp_minus_inf(env, ew, sw) : msat_make_fp_plus_inf(env, ew, sw);
  check_msat_error(t);

  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm)
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

  smt_sort *s = mk_sort(SMT_SORT_FLOATBV_RM);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_typecast_from_bvfloat(const typecast2t &cast)
{
  // Rounding mode symbol
  smt_astt rm_const;

  smt_astt from = convert_ast(cast.from);
  const mathsat_smt_ast *mfrom = mathsat_ast_downcast(from);

  msat_term t;
  smt_sort *s = nullptr;
  if(is_bv_type(cast.type)) {
    s = mk_sort(SMT_SORT_BV);

    // Conversion from float to integers always truncate, so we assume
    // the round mode to be toward zero
    rm_const = mk_smt_bvfloat_rm(ieee_floatt::ROUND_TO_ZERO);
    const mathsat_smt_ast *mrm = mathsat_ast_downcast(rm_const);

    t = msat_make_fp_to_bv(env, cast.type->get_width(), mrm->t, mfrom->t);
  } else if(is_floatbv_type(cast.type)) {
    unsigned ew = to_floatbv_type(cast.type).exponent;
    unsigned sw = to_floatbv_type(cast.type).fraction;

    s = mk_sort(SMT_SORT_FLOATBV, ew, sw);

    // Use the round mode
    rm_const = convert_rounding_mode(cast.rounding_mode);
    const mathsat_smt_ast *mrm = mathsat_ast_downcast(rm_const);

    t = msat_make_fp_cast(env, ew, sw, mrm->t, mfrom->t);
  }
  else
    abort();

  check_msat_error(t);
  assert(s != nullptr);

  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_typecast_to_bvfloat(const typecast2t &cast)
{
  smt_astt rm = convert_rounding_mode(cast.rounding_mode);
  const mathsat_smt_ast *mrm = mathsat_ast_downcast(rm);

  smt_astt from = convert_ast(cast.from);
  const mathsat_smt_ast *mfrom = mathsat_ast_downcast(from);

  unsigned ew = to_floatbv_type(cast.type).exponent;
  unsigned sw = to_floatbv_type(cast.type).fraction;
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);

  msat_term t;
  if(is_bool_type(cast.from))
  {
    // For bools, there is no direct conversion, so the cast is
    // transformed into fpa = b ? 1 : 0;
    const smt_ast *args[3];
    args[0] = from;
    args[1] = convert_ast(gen_one(cast.type));
    args[2] = convert_ast(gen_zero(cast.type));

    return mk_func_app(s, SMT_FUNC_ITE, args, 3);
  }

  if(is_unsignedbv_type(cast.from))
    t = msat_make_fp_from_ubv(env, ew, sw, mrm->t, mfrom->t);

  if(is_signedbv_type(cast.from))
    t = msat_make_fp_from_sbv(env, ew, sw, mrm->t, mfrom->t);

  if(is_floatbv_type(cast.from))
    t = msat_make_fp_cast(env, ew, sw, mrm->t, mfrom->t);

  check_msat_error(t);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_nearbyint_from_float(const nearbyint2t& expr)
{
  // Rounding mode symbol
  smt_astt rm = convert_rounding_mode(expr.rounding_mode);
  const mathsat_smt_ast *mrm = mathsat_ast_downcast(rm);

  smt_astt from = convert_ast(expr.from);
  const mathsat_smt_ast *mfrom = mathsat_ast_downcast(from);

  // Conversion from float, using the correct rounding mode
  msat_term t = msat_make_fp_round_to_int(env, mrm->t, mfrom->t);
  check_msat_error(t);

  smt_sortt s = convert_sort(expr.type);
  return new mathsat_smt_ast(this, s, t);
}

smt_astt mathsat_convt::mk_smt_bvfloat_arith_ops(const expr2tc& expr)
{
  // Rounding mode symbol
  smt_astt rm = convert_rounding_mode(*expr->get_sub_expr(0));
  const mathsat_smt_ast *mrm = mathsat_ast_downcast(rm);

  // Sides
  smt_astt s1 = convert_ast(*expr->get_sub_expr(1));
  const mathsat_smt_ast *ms1 = mathsat_ast_downcast(s1);

  msat_term t;
  if(is_ieee_sqrt2t(expr))
  {
    t = msat_make_fp_sqrt(env, mrm->t, ms1->t);
  }
  else
  {
    smt_astt s2 = convert_ast(*expr->get_sub_expr(2));
    const mathsat_smt_ast *ms2 = mathsat_ast_downcast(s2);

    switch (expr->expr_id)
    {
      case expr2t::ieee_add_id:
        t = msat_make_fp_plus(env, mrm->t, ms1->t, ms2->t);
        break;
      case expr2t::ieee_sub_id:
        t = msat_make_fp_minus(env, mrm->t, ms1->t, ms2->t);
        break;
      case expr2t::ieee_mul_id:
        t = msat_make_fp_times(env, mrm->t, ms1->t, ms2->t);
        break;
      case expr2t::ieee_div_id:
        t = msat_make_fp_div(env, mrm->t, ms1->t, ms2->t);
        break;
      case expr2t::ieee_fma_id:
      {
        // Mathsat doesn't support fma for now, if we force
        // the multiplication, it will provide the wrong answer
        std::cerr << "Mathsat doesn't support the fused multiply-add "
            "(fp.fma) operator" << std::endl;
      }
      default:
        abort();
    }
  }
  check_msat_error(t);

  unsigned ew = to_floatbv_type(expr->type).exponent;
  unsigned sw = to_floatbv_type(expr->type).fraction;
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);

  return new mathsat_smt_ast(this, s, t);
}

smt_ast *
mathsat_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new mathsat_smt_ast(this, s, (val) ? msat_make_true(env)
                                      : msat_make_false(env));
}

smt_ast *
mathsat_convt::mk_array_symbol(const std::string &name, const smt_sort *s,
                               smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *
mathsat_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  const mathsat_smt_sort *ms = mathsat_sort_downcast(s);
  // XXX - does 'd' leak?
  msat_decl d = msat_declare_function(env, name.c_str(), ms->t);
  assert(!MSAT_ERROR_DECL(d) && "Invalid function symbol declaration sort");

  msat_term t = msat_make_constant(env, d);
  check_msat_error(t);
  return new mathsat_smt_ast(this, s, t);
}

smt_sort *
mathsat_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_extract(const smt_ast *a, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);

  // If it's a floatbv, convert it to bv
  if(a->sort->id == SMT_SORT_FLOATBV)
  {
    msat_term t = msat_make_fp_as_ieeebv(env, mast->t);
    check_msat_error(t);

    smt_ast * bv = new mathsat_smt_ast(this, s, t);
    mast = mathsat_ast_downcast(bv);
  }

  msat_term t = msat_make_bv_extract(env, high, low, mast->t);
  check_msat_error(t);

  return new mathsat_smt_ast(this, s, t);
}

const smt_ast *
mathsat_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void
mathsat_convt::add_array_constraints_for_solving()
{
}

void
mathsat_convt::push_array_ctx()
{
}

void
mathsat_convt::pop_array_ctx()
{
}

const smt_ast* mathsat_smt_ast::select(smt_convt* ctx, const expr2tc& idx) const
{
  const smt_ast *args[2];
  args[0] = this;
  args[1] = ctx->convert_ast(idx);
  const smt_sort *rangesort = mathsat_sort_downcast(sort)->rangesort;
  return ctx->mk_func_app(rangesort, SMT_FUNC_SELECT, args, 2);
}

void mathsat_smt_ast::dump() const
{
  std::cout << msat_to_smtlib2(*_env, t) << std::endl;
}

size_t
mathsat_convt::get_exp_width(smt_sortt sort)
{
  const mathsat_smt_sort *ms = mathsat_sort_downcast(sort);
  size_t exp_width, mant_width;
  int ret = msat_is_fp_type(env, ms->t, &exp_width, &mant_width);
  assert(ret != 0 && "Non FP type passed to mathsat_convt::get_exp_width");
  return exp_width;
}

size_t
mathsat_convt::get_mant_width(smt_sortt sort)
{
  const mathsat_smt_sort *ms = mathsat_sort_downcast(sort);
  size_t exp_width, mant_width;
  int ret = msat_is_fp_type(env, ms->t, &exp_width, &mant_width);
  assert(ret != 0 && "Non FP type passed to mathsat_convt::get_mant_width");
  return mant_width;
}

mathsat_smt_ast::~mathsat_smt_ast()
{
  // We don't need to free the AST or the sort,
  // as freeing env does exactly the same
}

void mathsat_convt::dump_smt()
{
  print_mathsat_formula();
}
