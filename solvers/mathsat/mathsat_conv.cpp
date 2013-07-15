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
mathsat_convt::assert_lit(const literalt &l __attribute__((unused)))
{
  abort();
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
mathsat_convt::mk_func_app(const smt_sort *s __attribute__((unused)), smt_func_kind k __attribute__((unused)),
                             const smt_ast * const *args __attribute__((unused)),
                             unsigned int numargs __attribute__((unused)))
{
  abort();
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
mathsat_convt::mk_lit(const smt_ast *s __attribute__((unused)))
{
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
mathsat_convt::mk_smt_bvint(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)),
                              unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new mathsat_smt_ast(s, (val) ? msat_make_true(env)
                                      : msat_make_false(env));
}

smt_ast *
mathsat_convt::mk_smt_symbol(const std::string &name __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
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
