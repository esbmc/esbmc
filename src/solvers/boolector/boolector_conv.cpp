#include <boolector_conv.h>
#include <cstring>

extern "C" {
#include <btorcore.h>
}

smt_convt *create_new_boolector_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api __attribute__((unused)),
  array_iface **array_api,
  fp_convt **fp_api)
{
  boolector_convt *conv = new boolector_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

boolector_convt::boolector_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns), array_iface(false, false), fp_convt(this)
{
  if(int_encoding)
  {
    std::cerr << "Boolector does not support integer encoding mode"
              << std::endl;
    abort();
  }

  btor = boolector_new();
  boolector_set_opt(btor, BTOR_OPT_MODEL_GEN, 1);
  boolector_set_opt(btor, BTOR_OPT_AUTO_CLEANUP, 1);
}

boolector_convt::~boolector_convt()
{
  boolector_delete(btor);
  btor = nullptr;
}

smt_convt::resultt boolector_convt::dec_solve()
{
  pre_solve();

  int result = boolector_sat(btor);

  if(result == BOOLECTOR_SAT)
    return P_SATISFIABLE;

  if(result == BOOLECTOR_UNSAT)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string boolector_convt::solver_text()
{
  std::string ss = "Boolector ";
  ss += btor_version(btor);
  return ss;
}

void boolector_convt::assert_ast(const smt_ast *a)
{
  boolector_assert(btor, to_solver_smt_ast<btor_smt_ast>(a)->a);
}

smt_astt boolector_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  return new_ast(
    boolector_add(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_sub(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_mul(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_srem(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_urem(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_sdiv(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_udiv(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return fix_up_shift(boolector_sll, a, b, a->sort);
}

smt_astt boolector_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return fix_up_shift(boolector_sra, a, b, a->sort);
}

smt_astt boolector_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return fix_up_shift(boolector_srl, a, b, a->sort);
}

smt_astt boolector_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    boolector_neg(btor, to_solver_smt_ast<btor_smt_ast>(a)->a), a->sort);
}

smt_astt boolector_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    boolector_not(btor, to_solver_smt_ast<btor_smt_ast>(a)->a), a->sort);
}

smt_astt boolector_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_xnor(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_nor(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_nand(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_xor(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_or(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_and(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort);
}

smt_astt boolector_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    boolector_implies(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    boolector_xor(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    boolector_or(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    boolector_and(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    boolector_not(btor, to_solver_smt_ast<btor_smt_ast>(a)->a), boolean_sort);
}

smt_astt boolector_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_ult(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_slt(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_ugt(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_sgt(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_ulte(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_slte(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_ugte(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_sgte(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_eq(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    boolector_ne(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt boolector_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    boolector_write(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a,
      to_solver_smt_ast<btor_smt_ast>(c)->a),
    a->sort);
}

smt_astt boolector_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    boolector_read(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    a->sort->get_range_sort());
}

smt_ast *boolector_convt::mk_smt_int(
  const mp_integer &theint __attribute__((unused)),
  bool sign __attribute__((unused)))
{
  std::cerr << "Boolector can't create integer sorts" << std::endl;
  abort();
}

smt_ast *boolector_convt::mk_smt_real(const std::string &str
                                      __attribute__((unused)))
{
  std::cerr << "Boolector can't create Real sorts" << std::endl;
  abort();
}

smt_astt boolector_convt::mk_smt_bv(const mp_integer &theint, smt_sortt s)
{
  return new_ast(
    boolector_const(btor, integer2binary(theint, s->get_data_width()).c_str()),
    s);
}

smt_ast *boolector_convt::mk_smt_bool(bool val)
{
  BoolectorNode *node = (val) ? boolector_true(btor) : boolector_false(btor);
  const smt_sort *sort = boolean_sort;
  return new_ast(node, sort);
}

smt_ast *boolector_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *
boolector_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  symtable_type::iterator it = symtable.find(name);
  if(it != symtable.end())
    return it->second;

  BoolectorNode *node;

  switch(s->id)
  {
  case SMT_SORT_BV:
  case SMT_SORT_FIXEDBV:
  case SMT_SORT_BVFP:
  case SMT_SORT_BVFP_RM:
    node = boolector_var(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_BOOL:
    node = boolector_var(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_ARRAY:
    node = boolector_array(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  default:
    std::cerr << "Unknown type for symbol\n";
    abort();
  }

  btor_smt_ast *ast = new_ast(node, s);

  symtable.insert(symtable_type::value_type(name, ast));
  return ast;
}

smt_astt boolector_convt::mk_extract(
  const smt_ast *a,
  unsigned int high,
  unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  BoolectorNode *b = boolector_slice(btor, ast->a, high, low);
  return new_ast(b, s);
}

smt_astt boolector_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  BoolectorNode *b = boolector_sext(btor, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt boolector_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  BoolectorNode *b = boolector_uext(btor, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt boolector_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    boolector_concat(
      btor,
      to_solver_smt_ast<btor_smt_ast>(a)->a,
      to_solver_smt_ast<btor_smt_ast>(b)->a),
    s);
}

smt_astt boolector_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  return new_ast(
    boolector_cond(
      btor,
      to_solver_smt_ast<btor_smt_ast>(cond)->a,
      to_solver_smt_ast<btor_smt_ast>(t)->a,
      to_solver_smt_ast<btor_smt_ast>(f)->a),
    t->sort);
}

bool boolector_convt::get_bool(const smt_ast *a)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  const char *result = boolector_bv_assignment(btor, ast->a);

  assert(result != NULL && "Boolector returned null bv assignment string");

  bool res;
  switch(*result)
  {
  case '1':
    res = true;
    break;
  case '0':
    res = false;
    break;
  default:
    abort();
  }

  boolector_free_bv_assignment(btor, result);
  return res;
}

BigInt boolector_convt::get_bv(smt_astt a)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);

  const char *result = boolector_bv_assignment(btor, ast->a);
  BigInt val = string2integer(result, 2);
  boolector_free_bv_assignment(btor, result);

  return val;
}

expr2tc boolector_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(array);

  int size;
  char **indicies, **values;
  boolector_array_assignment(btor, ast->a, &indicies, &values, &size);

  BigInt val = 0;
  if(size > 0)
  {
    for(int i = 0; i < size; i++)
    {
      auto idx = string2integer(indicies[i], 2);
      if(idx.to_uint64() == index)
      {
        val = string2integer(values[i], 2);
        break;
      }
    }

    boolector_free_array_assignment(btor, indicies, values, size);
    return build_bv(subtype, val);
  }

  return gen_zero(subtype);
}

const smt_ast *boolector_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);

  const btor_smt_ast *side1 =
    to_solver_smt_ast<btor_smt_ast>(convert_ast(opers.side_1));
  const btor_smt_ast *side2 =
    to_solver_smt_ast<btor_smt_ast>(convert_ast(opers.side_2));

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed =
    (is_signedbv_type(opers.side_1) || is_signedbv_type(opers.side_2));

  BoolectorNode *res;
  if(is_add2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_saddo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_uaddo(btor, side1->a, side2->a);
    }
  }
  else if(is_sub2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_ssubo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_usubo(btor, side1->a, side2->a);
    }
  }
  else if(is_mul2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_smulo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_umulo(btor, side1->a, side2->a);
    }
  }
  else if(is_div2t(overflow.operand) || is_modulus2t(overflow.operand))
  {
    res = boolector_sdivo(btor, side1->a, side2->a);
  }
  else
  {
    return smt_convt::overflow_arith(expr);
  }

  const smt_sort *s = boolean_sort;
  return new_ast(res, s);
}

const smt_ast *
boolector_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void boolector_convt::add_array_constraints_for_solving()
{
}

void boolector_convt::push_array_ctx()
{
}

void boolector_convt::pop_array_ctx()
{
}

smt_ast *boolector_convt::fix_up_shift(
  shift_func_ptr fptr,
  smt_astt op0,
  smt_astt op1,
  smt_sortt res_sort)
{
  BoolectorNode *data_op, *shift_amount;
  bool need_to_shift_down = false;
  unsigned int bwidth;

  data_op = to_solver_smt_ast<btor_smt_ast>(op0)->a;
  bwidth = log2(op0->sort->get_data_width());

  // If we're a non-power-of-x number, some zero extension has to occur
  if(pow(2.0, bwidth) < op0->sort->get_data_width())
  {
    // Zero extend up to bwidth + 1
    bwidth++;
    unsigned int new_size = pow(2.0, bwidth);
    smt_astt zeroext = mk_zero_ext(op0, new_size - op0->sort->get_data_width());
    data_op = to_solver_smt_ast<btor_smt_ast>(zeroext)->a;
    need_to_shift_down = true;
  }

  // We also need to reduce the shift-amount operand down to log2(data_op) len
  shift_amount = boolector_slice(
    btor, to_solver_smt_ast<btor_smt_ast>(op1)->a, bwidth - 1, 0);

  BoolectorNode *shift = fptr(btor, data_op, shift_amount);

  // If zero extension occurred, cut off the top few bits of this value.
  if(need_to_shift_down)
    shift = boolector_slice(btor, shift, res_sort->get_data_width() - 1, 0);

  return new_ast(shift, res_sort);
}

void boolector_convt::dump_smt()
{
  boolector_dump_smt2(btor, stdout);
}

void btor_smt_ast::dump() const
{
  boolector_dump_smt2_node(boolector_get_btor(a), stdout, a);
}

void boolector_convt::print_model()
{
  boolector_print_model(btor, const_cast<char *>("smt2"), stdout);
}

smt_sortt boolector_convt::mk_bool_sort()
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_BOOL, boolector_bool_sort(btor), 1);
}

smt_sortt boolector_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_BV, boolector_bitvec_sort(btor, width), width);
}

smt_sortt boolector_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_FIXEDBV, boolector_bitvec_sort(btor, width), width);
}

smt_sortt boolector_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<BoolectorSort>(domain);
  auto range_sort = to_solver_smt_sort<BoolectorSort>(range);

  auto t = boolector_array_sort(btor, domain_sort->s, range_sort->s);
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_ARRAY, t, domain_sort->get_data_width(), range);
}

smt_sortt boolector_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_BVFP,
    boolector_bitvec_sort(btor, ew + sw + 1),
    ew + sw + 1,
    sw + 1);
}

smt_sortt boolector_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_BVFP_RM, boolector_bitvec_sort(btor, 3), 3);
}
