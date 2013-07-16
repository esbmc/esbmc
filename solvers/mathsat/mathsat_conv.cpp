#include "mathsat_conv.h"

prop_convt *
create_new_mathsat_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
    return new mathsat_convt(is_cpp, int_encoding, ns);
}

mathsat_convt::mathsat_convt(bool is_cpp, bool int_encoding,
                             const namespacet &ns)
  : smt_convt(true, int_encoding, ns, is_cpp, false, true, true)
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
mathsat_convt::assert_lit(const literalt &l)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(lit_to_ast(l));
  msat_assert_formula(env, mast->t);
}

prop_convt::resultt
mathsat_convt::dec_solve()
{
  abort();
}

expr2tc
mathsat_convt::get(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

tvt
mathsat_convt::l_get(literalt l __attribute__((unused)))
{
  abort();
}

const std::string
mathsat_convt::solver_text()
{
  abort();
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
    r = msat_make_equal(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_NOT:
    r = msat_make_not(env, args[0]->t);
    break;
  case SMT_FUNC_AND:
    r = msat_make_and(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVADD:
    r = msat_make_bv_plus(env, args[0]->t, args[1]->t);
    break;
  case SMT_FUNC_BVSUB:
    r = msat_make_bv_minus(env, args[0]->t, args[1]->t);
    break;
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
    abort();
  }

  return new mathsat_smt_ast(s, r);
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
    return new mathsat_smt_sort(k, msat_get_bv_type(env, uint));
  case SMT_SORT_ARRAY:
  {
    dom = va_arg(ap, const mathsat_smt_sort *);
    range = va_arg(ap, const mathsat_smt_sort *);
    mathsat_smt_sort *result =
      new mathsat_smt_sort(k, msat_get_array_type(env, dom->t, range->t));
    size_t sz = 0;
    int tmp;
    tmp = msat_is_bv_type(env, dom->t, &sz);
    assert(tmp == 1 && "Domain of array must be a bitvector");
    result->array_dom_width = sz;
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

literalt
mathsat_convt::mk_lit(const smt_ast *a)
{
  const mathsat_smt_ast *mast = mathsat_ast_downcast(a);
  literalt l = new_variable();
  const mathsat_smt_ast *m2 = mathsat_ast_downcast(lit_to_ast(l));

  msat_term r = msat_make_equal(env, mast->t, m2->t);
  msat_assert_formula(env, r);
  return l;
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

  if (theint.is_negative())
    ss << theint.to_long();
  else
    ss << theint.to_ulong();

  std::string rep = ss.str();

  // Make bv int from textual representation.
  msat_term t = msat_make_bv_number(env, rep.c_str(), w, 10);
  assert(!MSAT_ERROR_TERM(t) && "Error creating mathsat BV integer term");
  smt_sort *s = mk_sort(SMT_SORT_BV, w, false);
  return new mathsat_smt_ast(s, t);
}

smt_ast *
mathsat_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new mathsat_smt_ast(s, (val) ? msat_make_true(env)
                                      : msat_make_false(env));
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
  return new mathsat_smt_ast(s, t);
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
mathsat_convt::mk_extract(const smt_ast *a __attribute__((unused)), unsigned int high __attribute__((unused)),
                            unsigned int low __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}
