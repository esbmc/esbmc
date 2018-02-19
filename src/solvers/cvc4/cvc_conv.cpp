#include <util/c_types.h>
#include <cvc_conv.h>

smt_convt *create_new_cvc_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api __attribute__((unused)),
  array_iface **array_api,
  fp_convt **fp_api)
{
  cvc_convt *conv = new cvc_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

cvc_convt::cvc_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns),
    array_iface(false, false),
    fp_convt(this),
    em(),
    smt(&em),
    sym_tab()
{
  // Already initialized stuff in the constructor list,

  smt.setOption("produce-models", true);
  smt.setLogic("QF_AUFBV");

  assert(!int_encoding && "Integer encoding mode for CVC unimplemented");
}

smt_convt::resultt cvc_convt::dec_solve()
{
  pre_solve();

  CVC4::Result r = smt.checkSat();
  if(r.isSat())
    return P_SATISFIABLE;

  if(!r.isUnknown())
    return P_ERROR;

  return P_UNSATISFIABLE;
}

bool cvc_convt::get_bool(const smt_ast *a)
{
  auto const *ca = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(a);
  CVC4::Expr e = smt.getValue(ca->a);
  return e.getConst<bool>();
}

BigInt cvc_convt::get_bv(smt_astt a)
{
  auto const *ca = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(a);
  CVC4::Expr e = smt.getValue(ca->a);
  CVC4::BitVector foo = e.getConst<CVC4::BitVector>();
  return BigInt(foo.toInteger().getUnsignedLong());
}

expr2tc cvc_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  auto const *carray = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(array);
  size_t orig_w = array->sort->get_domain_width();

  smt_astt tmpast = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(index), orig_w);
  auto const *tmpa = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(tmpast);
  CVC4::Expr e = em.mkExpr(CVC4::kind::SELECT, carray->a, tmpa->a);
  delete tmpast;

  solver_smt_ast<CVC4::Expr> *tmpb =
    new solver_smt_ast<CVC4::Expr>(this, convert_sort(subtype), e);
  expr2tc result = get_by_ast(subtype, tmpb);
  delete tmpb;

  return result;
}

const std::string cvc_convt::solver_text()
{
  std::stringstream ss;
  ss << "CVC " << CVC4::Configuration::getVersionString();
  return ss.str();
}

void cvc_convt::assert_ast(const smt_ast *a)
{
  auto const *ca = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(a);
  smt.assertFormula(ca->a);
}

smt_ast *cvc_convt::mk_func_app(
  const smt_sort *s,
  smt_func_kind k,
  const smt_ast *const *_args,
  unsigned int numargs)
{
  const solver_smt_ast<CVC4::Expr> *args[4];
  unsigned int i;

  assert(numargs <= 4);
  for(i = 0; i < numargs; i++)
    args[i] = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(_args[i]);

  CVC4::Expr e;

  switch(k)
  {
  case SMT_FUNC_EQ:
    e = em.mkExpr(CVC4::kind::EQUAL, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_NOTEQ:
    e = em.mkExpr(CVC4::kind::DISTINCT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_AND:
    e = em.mkExpr(CVC4::kind::AND, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_OR:
    e = em.mkExpr(CVC4::kind::OR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_XOR:
    e = em.mkExpr(CVC4::kind::XOR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_IMPLIES:
    e = em.mkExpr(CVC4::kind::IMPLIES, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_ITE:
    e = em.mkExpr(CVC4::kind::ITE, args[0]->a, args[1]->a, args[2]->a);
    break;
  case SMT_FUNC_NOT:
    e = em.mkExpr(CVC4::kind::NOT, args[0]->a);
    break;
  case SMT_FUNC_BVNOT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_NOT, args[0]->a);
    break;
  case SMT_FUNC_BVNEG:
    e = em.mkExpr(CVC4::kind::BITVECTOR_NEG, args[0]->a);
    break;
  case SMT_FUNC_BVADD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_PLUS, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSUB:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SUB, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVMUL:
    e = em.mkExpr(CVC4::kind::BITVECTOR_MULT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSDIV:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SDIV, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUDIV:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UDIV, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSMOD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SREM, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUMOD:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UREM, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVLSHR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_LSHR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVASHR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ASHR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSHL:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SHL, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUGT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UGT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVUGTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_UGE, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVULT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ULT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVULTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_ULE, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSGT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SGT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSGTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SGE, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSLT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SLT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVSLTE:
    e = em.mkExpr(CVC4::kind::BITVECTOR_SLE, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVAND:
    e = em.mkExpr(CVC4::kind::BITVECTOR_AND, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVOR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_OR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_BVXOR:
    e = em.mkExpr(CVC4::kind::BITVECTOR_XOR, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_STORE:
    e = em.mkExpr(CVC4::kind::STORE, args[0]->a, args[1]->a, args[2]->a);
    break;
  case SMT_FUNC_SELECT:
    e = em.mkExpr(CVC4::kind::SELECT, args[0]->a, args[1]->a);
    break;
  case SMT_FUNC_CONCAT:
    e = em.mkExpr(CVC4::kind::BITVECTOR_CONCAT, args[0]->a, args[1]->a);
    break;
  default:
    std::cerr << "Unimplemented SMT function \"" << smt_func_name_table[k]
              << "\" in CVC conversion" << std::endl;
    abort();
  }

  return new solver_smt_ast<CVC4::Expr>(this, s, e);
}

smt_ast *cvc_convt::mk_smt_int(
  const mp_integer &theint __attribute__((unused)),
  bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *cvc_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  abort();
}

smt_ast *cvc_convt::mk_smt_bv(smt_sortt s, const mp_integer &theint)
{
  std::size_t w = s->get_data_width();

  // Seems we can't make negative bitvectors; so just pull the value out and
  // assume CVC is going to cut the top off correctly.
  CVC4::BitVector bv = CVC4::BitVector(w, (unsigned long int)theint.to_int64());
  CVC4::Expr e = em.mkConst(bv);
  return new solver_smt_ast<CVC4::Expr>(this, s, e);
}

smt_ast *cvc_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = boolean_sort;
  CVC4::Expr e = em.mkConst(val);
  return new solver_smt_ast<CVC4::Expr>(this, s, e);
}

smt_ast *cvc_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype)
{
  return mk_smt_symbol(name, s);
}

smt_ast *cvc_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  // Standard arrangement: if we already have the name, return the expression
  // from the symbol table. If not, time for a new name.
  if(sym_tab.isBound(name))
  {
    CVC4::Expr e = sym_tab.lookup(name);
    return new solver_smt_ast<CVC4::Expr>(this, s, e);
  }

  // Time for a new one.
  CVC4::Expr e =
    em.mkVar(name, to_solver_smt_sort<CVC4::Type>(s)->s); // "global", eh?
  sym_tab.bind(name, e, true);
  return new solver_smt_ast<CVC4::Expr>(this, s, e);
}

smt_sort *cvc_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *cvc_convt::mk_extract(
  const smt_ast *a,
  unsigned int high,
  unsigned int low,
  const smt_sort *s)
{
  auto const *ca = to_solver_smt_ast<solver_smt_ast<CVC4::Expr>>(a);
  CVC4::BitVectorExtract ext(high, low);
  CVC4::Expr ext2 = em.mkConst(ext);
  CVC4::Expr fin = em.mkExpr(CVC4::Kind::BITVECTOR_EXTRACT, ext2, ca->a);
  return new solver_smt_ast<CVC4::Expr>(this, s, fin);
}

const smt_ast *
cvc_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void cvc_convt::add_array_constraints_for_solving()
{
}

void cvc_convt::push_array_ctx()
{
}

void cvc_convt::pop_array_ctx()
{
}

smt_sortt cvc_convt::mk_bool_sort()
{
  return new solver_smt_sort<CVC4::Type>(SMT_SORT_BOOL, em.booleanType(), 1);
}

smt_sortt cvc_convt::mk_bv_sort(const smt_sort_kind k, std::size_t width)
{
  return new solver_smt_sort<CVC4::Type>(k, em.mkBitVectorType(width), width);
}

smt_sortt cvc_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<CVC4::Type>(domain);
  auto range_sort = to_solver_smt_sort<CVC4::Type>(range);

  auto t = em.mkArrayType(domain_sort->s, range_sort->s);
  return new solver_smt_sort<CVC4::Type>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_sortt cvc_convt::mk_bv_fp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<CVC4::Type>(
    SMT_SORT_FAKE_FLOATBV,
    em.mkBitVectorType(ew + sw + 1),
    ew + sw + 1,
    sw + 1);
}

smt_sortt cvc_convt::mk_bv_fp_rm_sort()
{
  return new solver_smt_sort<CVC4::Type>(
    SMT_SORT_FAKE_FLOATBV_RM, em.mkBitVectorType(2), 2);
}
