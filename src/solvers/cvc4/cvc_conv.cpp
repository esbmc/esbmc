#include <c_types.h>

#include "cvc_conv.h"

smt_convt *
create_new_cvc_solver(bool int_encoding, const namespacet &ns,
                      const optionst &opts __attribute__((unused)),
                      tuple_iface **tuple_api __attribute__((unused)),
                      array_iface **array_api)
{
  cvc_convt *conv = new cvc_convt(int_encoding, ns);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

cvc_convt::cvc_convt(bool int_encoding, const namespacet &ns)
   : smt_convt(int_encoding, ns), array_iface(false, false),
     em(), smt(&em), sym_tab()
{
  // Already initialized stuff in the constructor list,

  smt.setOption("produce-models", true);
  smt.setLogic("QF_AUFBV");

  assert(!int_encoding && "Integer encoding mode for CVC unimplemented");
}

cvc_convt::~cvc_convt()
{
}

smt_convt::resultt
cvc_convt::dec_solve()
{
  pre_solve();

  CVC4::Result r = smt.checkSat();
  if (r.isSat())
    return P_SATISFIABLE;
  else if (!r.isUnknown())
    return P_UNSATISFIABLE;
  else {
    std::cerr << "Error solving satisfiability of formula with CVC"
              << std::endl;
    abort();
  }
}

tvt
cvc_convt::l_get(const smt_ast *a)
{
  const cvc_smt_ast *ca = cvc_ast_downcast(a);
  constant_bool2tc b = get_bool(ca);
  if (b->value)
    return tvt(true);
  else
    return tvt(false);
}

expr2tc
cvc_convt::get_bool(const smt_ast *a)
{
  const cvc_smt_ast *ca = cvc_ast_downcast(a);
  CVC4::Expr e = smt.getValue(ca->e);
  bool foo = e.getConst<bool>();
  return constant_bool2tc(foo);
}

expr2tc
cvc_convt::get_bv(const type2tc &t __attribute__((unused)), const smt_ast *a)
{
  const cvc_smt_ast *ca = cvc_ast_downcast(a);
  CVC4::Expr e = smt.getValue(ca->e);
  CVC4::BitVector foo = e.getConst<CVC4::BitVector>();
  // XXX, might croak on 32 bit machines. I'm not aware of a fixed-width api
  // for CVC right now.
  uint64_t val = foo.toInteger().getUnsignedLong();
  return constant_int2tc(get_uint_type(foo.getSize()), BigInt(val));
}

expr2tc
cvc_convt::get_array_elem(const smt_ast *array, uint64_t index,
                          const type2tc &elem_sort)
{
  const cvc_smt_ast *carray = cvc_ast_downcast(array);
  unsigned int orig_w = array->sort->domain_width;

  smt_ast *tmpast = mk_smt_bvint(BigInt(index), false, orig_w);
  const cvc_smt_ast *tmpa = cvc_ast_downcast(tmpast);
  CVC4::Expr e = em.mkExpr(CVC4::kind::SELECT, carray->e, tmpa->e);
  free(tmpast);

  cvc_smt_ast *tmpb = new cvc_smt_ast(this, convert_sort(elem_sort), e);
  expr2tc result = get_bv(type2tc(), tmpb);
  free(tmpb);

  return result;
}

const std::string
cvc_convt::solver_text()
{
  std::stringstream ss;
  ss << "CVC " << CVC4::Configuration::getVersionString();
  return ss.str();
}

void
cvc_convt::assert_ast(const smt_ast *a)
{
  const cvc_smt_ast *ca = cvc_ast_downcast(a);
  smt.assertFormula(ca->e);
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
    if (args[0]->sort->id == SMT_SORT_BOOL) {
      e = em.mkExpr(CVC4::kind::IFF, args[0]->e, args[1]->e);
    } else {
      e = em.mkExpr(CVC4::kind::EQUAL, args[0]->e, args[1]->e);
    }
    break;
  case SMT_FUNC_NOTEQ:
    e = em.mkExpr(CVC4::kind::DISTINCT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_AND:
    e = em.mkExpr(CVC4::kind::AND, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_OR:
    e = em.mkExpr(CVC4::kind::OR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_XOR:
    e = em.mkExpr(CVC4::kind::XOR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_IMPLIES:
    e = em.mkExpr(CVC4::kind::IMPLIES, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_ITE:
    e = em.mkExpr(CVC4::kind::ITE, args[0]->e, args[1]->e, args[2]->e);
    break;
  case SMT_FUNC_NOT:
    e = em.mkExpr(CVC4::kind::NOT, args[0]->e);
    break;
  case SMT_FUNC_BVNOT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_NOT, args[0]->e);
    break;
  case SMT_FUNC_BVNEG:
    e = em.mkExpr(CVC4::kind::BITVECTOR_NEG, args[0]->e);
    break;
  case SMT_FUNC_BVADD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_PLUS, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSUB:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SUB, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVMUL:
    e = em.mkExpr(CVC4::kind::BITVECTOR_MULT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSDIV:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SDIV, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVUDIV:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UDIV, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSMOD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SREM, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVUMOD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UREM, args[0]->e, args[1]->e);
  case SMT_FUNC_BVLSHR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_LSHR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVASHR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ASHR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSHL:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SHL, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVUGT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UGT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVUGTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UGE, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVULT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ULT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVULTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ULE, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSGT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SGT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSGTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SGE, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSLT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SLT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVSLTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SLE, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVAND:
    e = em.mkExpr(CVC4::kind::BITVECTOR_AND, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVOR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_OR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_BVXOR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_XOR, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_STORE:
    e = em.mkExpr(CVC4::kind::STORE, args[0]->e, args[1]->e, args[2]->e);
    break;
  case SMT_FUNC_SELECT:
    e = em.mkExpr(CVC4::kind::SELECT, args[0]->e, args[1]->e);
    break;
  case SMT_FUNC_CONCAT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_CONCAT, args[0]->e, args[1]->e);
    break;
  default:
    std::cerr << "Unimplemented SMT function \"" << smt_func_name_table[k]
              << "\" in CVC conversion" << std::endl;
    abort();
  }

  return new cvc_smt_ast(this, s, e);
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
    return new cvc_smt_sort(k, t, uint);
  }
  case SMT_SORT_ARRAY:
  {
    const cvc_smt_sort *dom = va_arg(ap, const cvc_smt_sort*);
    const cvc_smt_sort *range = va_arg(ap, const cvc_smt_sort*);
    CVC4::ArrayType t = em.mkArrayType(dom->t, range->t);
    return new cvc_smt_sort(k, t, range->data_width, dom->data_width);
  }
  case SMT_SORT_FLOATBV:
  {
    std::cerr << "CVC4 can't create floating point sorts" << std::endl;
    abort();
  }
  default:
    std::cerr << "Unimplemented smt sort " << k << " in CVC mk_sort"
              << std::endl;
    abort();
  }
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
  CVC4::BitVector bv = CVC4::BitVector(w, (uint64_t)theint.to_int64());
  CVC4::Expr e = em.mkConst(bv);
  return new cvc_smt_ast(this, s, e);
}

smt_ast* cvc_convt::mk_smt_bvfloat(const ieee_floatt &thereal,
                                   unsigned ew, unsigned sw)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_bvfloat_nan(unsigned ew, unsigned sw)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_typecast_from_bvfloat(const typecast2t& cast)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_typecast_to_bvfloat(const typecast2t& cast)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_bvfloat_arith_ops(const expr2tc& expr)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_astt cvc_convt::mk_smt_nearbyint_from_float(const nearbyint2t& expr)
{
  std::cerr << "CVC4 can't create floating point sorts" << std::endl;
  abort();
}

smt_ast *
cvc_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  CVC4::Expr e = em.mkConst(val);
  return new cvc_smt_ast(this, s, e);
}

smt_ast *
cvc_convt::mk_array_symbol(const std::string &name, const smt_sort *s,
                           smt_sortt array_subtype)
{
  return mk_smt_symbol(name, s);
}

smt_ast *
cvc_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  const cvc_smt_sort *sort = cvc_sort_downcast(s);

  // If someone's making a tuple-symbol, wave our hands and do nothing. It's
  // the tuple modelling code doing some symbol sillyness.
  if (s->id == SMT_SORT_STRUCT || s->id == SMT_SORT_UNION)
    return NULL;

  // Standard arrangement: if we already have the name, return the expression
  // from the symbol table. If not, time for a new name.
  if (sym_tab.isBound(name)) {
    CVC4::Expr e = sym_tab.lookup(name);
    return new cvc_smt_ast(this, s, e);
  }

  // Time for a new one.
  CVC4::Expr e = em.mkVar(name, sort->t); // "global", eh?
  sym_tab.bind(name, e, true);
  return new cvc_smt_ast(this, s, e);
}

smt_sort *
cvc_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
cvc_convt::mk_extract(const smt_ast *a, unsigned int high,
                            unsigned int low, const smt_sort *s)
{
  const cvc_smt_ast *ca = cvc_ast_downcast(a);
  CVC4::BitVectorExtract ext(high, low);
  CVC4::Expr ext2 = em.mkConst(ext);
  CVC4::Expr fin = em.mkExpr(CVC4::Kind::BITVECTOR_EXTRACT, ext2, ca->e);
  return new cvc_smt_ast(this, s, fin);
}

const smt_ast *
cvc_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void
cvc_convt::add_array_constraints_for_solving()
{
  return;
}

void
cvc_convt::push_array_ctx(void)
{
  return;
}

void
cvc_convt::pop_array_ctx(void)
{
  return;
}
