#include <string>
#include <sstream>

#include "mathsat_conv.h"

// :|
#include <gmp.h>

#include <ansi-c/c_types.h>

smt_convt *
create_new_mathsat_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                          const optionst &opts __attribute__((unused)),
                          tuple_iface **tuple_api __attribute__((unused)),
                          array_iface **array_api)
{
  mathsat_convt *conv = new mathsat_convt(is_cpp, int_encoding, ns);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

mathsat_convt::mathsat_convt(bool is_cpp, bool int_encoding,
                             const namespacet &ns)
  : smt_convt(int_encoding, ns, is_cpp), array_iface(false, false)
{

  if (int_encoding) {
    std::cerr << "MathSAT converter doesn't support integer encoding"
              << std::endl;
    abort();
  }

  cfg = msat_create_config();
  /* XXX -- where is the list of options?" */
  msat_set_option(cfg, "model_generation", "true");
  env = msat_create_env(cfg);
}

mathsat_convt::~mathsat_convt(void)
{
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

  if (msat_term_is_true(env, t)) {
    return true_expr;
  } else if (msat_term_is_false(env, t)) {
    return false_expr;
  } else {
    std::cerr << "Boolean model value is neither true or false" << std::endl;
    abort();
  }
}

expr2tc
mathsat_convt::get_bv(const type2tc &_t __attribute__((unused)),
                      const smt_ast *a)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  msat_term t = msat_get_model_value(env, mast->t);
  assert(msat_term_is_number(env, t) && "Model value of bitvector isn't "
         "a bitvector");

  // GMP rational value object. Mildly irritating that we need to use GMP
  // directly; particularly seeing how a version incompatibility between
  // MathSAT and the local version of GMP is now fatal.
  mpq_t val;
  mpq_init(val);
  if (msat_term_to_number(env, t, val)) {
    std::cerr << "Error fetching number from MathSAT. Message reads:"
              << std::endl;
    std::cerr << "\"" << msat_last_error_message(env) << "\""
              << std::endl;
    abort();
  }

  mpz_t num;
  mpz_init(num);
  mpz_set(num, mpq_numref(val));
  char buffer[mpz_sizeinbase(num, 10) + 2];
  mpz_get_str(buffer, 10, num);
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
  free(tmpast);

  mathsat_smt_ast *tmpb = new mathsat_smt_ast(this, convert_sort(elem_sort), t);
  expr2tc result = get_bv(type2tc(), tmpb);
  free(tmpb);

  return result;
}

tvt
mathsat_convt::l_get(const smt_ast *a)
{
  constant_bool2tc b = get_bool(a);
  if (b->constant_value)
    return tvt(true);
  else if (!b->constant_value)
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
  s = s;
  k = k;
  numargs = numargs;

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
    // Another thing that mathsat doesn't implement for no reasons.
    // Do this as and(or(a,b),not(and(a,b)))
    msat_term and2 = msat_make_and(env, args[0]->t, args[1]->t);
    msat_term notand2 = msat_make_not(env, and2);
    msat_term or1 = msat_make_or(env, args[0]->t, args[1]->t);
    r = msat_make_and(env, or1, notand2);
    break;
  }
  case SMT_FUNC_IMPLIES:
    // MathSAT doesn't seem to implement this; so do it manually. Following the
    // CNF conversion CBMC does, this is: lor(lnot(a), b)
    r = msat_make_not(env, args[0]->t);
    r = msat_make_or(env, r, args[1]->t);
    break;
  case SMT_FUNC_ITE:
    if (s->id == SMT_SORT_BOOL) {
      // Once more, MathSAT shows a peculiar dislike of implementing the
      // simplest thing with booleans. Follow CBMC's CNF flattening and make
      // this (with c = cond, t = trueval, f = falseval):
      //
      //   or(and(c,t),and(not(c), f))
      msat_term land1 = msat_make_and(env, args[0]->t, args[1]->t);
      msat_term notval = msat_make_not(env, args[0]->t);
      msat_term land2 = msat_make_and(env, notval, args[2]->t);
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
  case SMT_FUNC_BVNEG:
    r = msat_make_bv_neg(env, args[0]->t);
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
  case SMT_FUNC_BVADD:
    r = msat_make_bv_plus(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSUB:
    r = msat_make_bv_minus(env, args[0]->t, args[1]->t);
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
  case SMT_FUNC_BVSGTE:
  {
    // This is !SLT
    assert(s->id == SMT_SORT_BOOL);
    const smt_ast *a = mk_func_app(s, SMT_FUNC_BVSLT, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_BVSGT:
  {
    // This is !SLTE
    assert(s->id == SMT_SORT_BOOL);
    const smt_ast *a = mk_func_app(s, SMT_FUNC_BVSLTE, _args, numargs);
    return mk_func_app(s, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_BVSLTE:
    r = msat_make_bv_sleq(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSLT:
    r = msat_make_bv_slt(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_STORE:
    r = msat_make_array_write(env, args[0]->t, args[1]->t, args[2]->t);
    break;
  case SMT_FUNC_SELECT:
    r = msat_make_array_read(env, args[0]->t, args[1]->t);
    break;

  default:
    std::cerr << "Unhandled SMT function \"" << smt_func_name_table[k] << "\" "
              << "in mathsat conversion" << std::endl;
    abort();
  }

  if (MSAT_ERROR_TERM(r)) {
    std::cerr << "Error creating SMT " << smt_func_name_table[k] << " function "
              << "application" << std::endl;
    std::cerr << "Error text: \"" << msat_last_error_message(env) << "\""
              << std::endl;
    abort();
  }

  return new mathsat_smt_ast(this, s, r);
}

smt_sort *
mathsat_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;
  unsigned long uint;
  int thebool;
  const mathsat_smt_sort *dom, *range;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
  case SMT_SORT_REAL:
    std::cerr << "Sorry, no integer encoding sorts for MathSAT" << std::endl;
    abort();
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    thebool = thebool;
    return new mathsat_smt_sort(k, msat_get_bv_type(env, uint), uint);
  case SMT_SORT_ARRAY:
  {
    dom = va_arg(ap, const mathsat_smt_sort *);
    range = va_arg(ap, const mathsat_smt_sort *);
    mathsat_smt_sort *result =
      new mathsat_smt_sort(k, msat_get_array_type(env, dom->t, range->t),
                           range->data_width, dom->data_width);
    size_t sz = 0;
    int tmp;
    tmp = msat_is_bv_type(env, dom->t, &sz);
    assert(tmp == 1 && "Domain of array must be a bitvector");
    return result;
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
mathsat_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_bvint(const mp_integer &theint,
                            bool sign __attribute__((unused)), unsigned int w)
{
  std::stringstream ss;

  // MathSAT refuses to parse negative integers. So, feed it binary.
  std::string str = integer2binary(theint, w);

  // Make bv int from textual representation.
  msat_term t = msat_make_bv_number(env, str.c_str(), w, 2);
  assert(!MSAT_ERROR_TERM(t) && "Error creating mathsat BV integer term");
  smt_sort *s = mk_sort(SMT_SORT_BV, w, false);
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
  assert(!MSAT_ERROR_TERM(t) && "Invalid function decl for mathsat term");
  return new mathsat_smt_ast(this, s, t);
}

smt_sort *
mathsat_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_sort *
mathsat_convt::mk_union_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_extract(const smt_ast *a, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  msat_term t = msat_make_bv_extract(env, high, low, mast->t);
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
  return;
}

void
mathsat_convt::push_array_ctx(void)
{
  return;
}

void
mathsat_convt::pop_array_ctx(void)
{
  return;
}
