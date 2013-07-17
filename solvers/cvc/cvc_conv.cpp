#include "cvc_conv.h"

prop_convt *
create_new_cvc_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
    return new cvc_convt(is_cpp, int_encoding, ns);
}

cvc_convt::cvc_convt(bool is_cpp, bool int_encoding, const namespacet &ns)
   : smt_convt(true, int_encoding, ns, is_cpp, false, true, false),
     em(), smt(&em)
{
  // Already initialized stuff in the constructor list,

  assert(!int_encoding && "Integer encoding mode for CVC unimplemented");

  smt_post_init();
}

cvc_convt::~cvc_convt()
{
  abort();
}

prop_convt::resultt
cvc_convt::dec_solve()
{
  abort();
}

expr2tc
cvc_convt::get(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

tvt
cvc_convt::l_get(literalt l __attribute__((unused)))
{
  abort();
}

const std::string
cvc_convt::solver_text()
{
  abort();
}

void
cvc_convt::assert_lit(const literalt &l __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_func_app(const smt_sort *s, smt_func_kind k,
                             const smt_ast * const *_args,
                             unsigned int numargs)
{
  const cvc_smt_ast *args[4];
  unsigned int i;

  assert(numargs <= 4);
  for (i = 0; i < numargs; i++)
    args[i] = cvc_ast_downcast(_args[i]);

  CVC4::Expr e;

  switch (k) {
  case SMT_FUNC_EQ:
    e = em.mkExpr(CVC4::kind::EQUAL, args[0]->e, args[1]->e);
    break;
  default:
    std::cerr << "Unimplemented SMT function \"" << smt_func_name_table[k]
              << "\" in CVC conversion" << std::endl;
    abort();
  }

  return new cvc_smt_ast(s, e);
}

smt_sort *
cvc_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;
  unsigned long uint;
  int thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_BOOL:
  {
    CVC4::BooleanType t = em.booleanType();
    return new cvc_smt_sort(k, t);
  }
  case SMT_SORT_BV:
  {
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    thebool = thebool;
    CVC4::BitVectorType t = em.mkBitVectorType(uint);
    return new cvc_smt_sort(k, t);
  }
  case SMT_SORT_ARRAY:
  {
    const cvc_smt_sort *dom = va_arg(ap, const cvc_smt_sort*);
    const cvc_smt_sort *range = va_arg(ap, const cvc_smt_sort*);
    CVC4::ArrayType t = em.mkArrayType(dom->t, range->t);
    return new cvc_smt_sort(k, t);
  }
  default:
    std::cerr << "Unimplemented smt sort " << k << " in CVC mk_sort"
              << std::endl;
    abort();
  }
}

literalt
cvc_convt::mk_lit(const smt_ast *a)
{
  const cvc_smt_ast *cast = cvc_ast_downcast(a);
  literalt l = new_variable();
  const cvc_smt_ast *c2 = cvc_ast_downcast(lit_to_ast(l));

  CVC4::Expr e = em.mkExpr(CVC4::Kind::EQUAL, cast->e, c2->e);
  smt.assertFormula(e);
  return l;
}

smt_ast *
cvc_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w)
{
  const smt_sort *s = mk_sort(SMT_SORT_BV, w, false);

  // Seems we can't make negative bitvectors; so just pull the value out and
  // assume CVC is going to cut the top off correctly.
  CVC4::BitVector bv = CVC4::BitVector(w, (uint64_t)theint.to_ulong());
  CVC4::Expr e = em.mkConst(bv);
  return new cvc_smt_ast(s, e);
}

smt_ast *
cvc_convt::mk_smt_bool(bool val __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  const cvc_smt_sort *sort = cvc_sort_downcast(s);
  CVC4::Expr e = em.mkVar(name, sort->t); // "global", eh?
  return new cvc_smt_ast(s, e);
}

smt_sort *
cvc_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_sort *
cvc_convt::mk_union_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_extract(const smt_ast *a __attribute__((unused)), unsigned int high __attribute__((unused)),
                            unsigned int low __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}
