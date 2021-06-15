#include <bitwuzla_conv.h>
#include <cstring>

#define new_ast new_solver_ast<bitw_smt_ast>

void error_handler(const char *msg)
{
  std::cerr << "Bitwuzla error encountered\n";
  std::cerr << msg << '\n';
  abort();
}


smt_convt *create_new_bitwuzla_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api [[gnu::unused]],
  array_iface **array_api,
  fp_convt **fp_api)
{
  //std::cerr << "Bitwuzla support has not been yet implemented\n";
  //abort();
  bitwuzla_convt *conv = new bitwuzla_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}


bitwuzla_convt::bitwuzla_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns), array_iface(true, true), fp_convt(this)
{
  /*
  if(int_encoding)
  {
    std::cerr << "Boolector does not support integer encoding mode"
              << std::endl;
    abort();
  }
  
  btor = boolector_new();
  boolector_set_opt(btor, BTOR_OPT_MODEL_GEN, 1);
  boolector_set_opt(btor, BTOR_OPT_AUTO_CLEANUP, 1);
  boolector_set_abort(error_handler);
  */
  bitw = bitwuzla_new();
  bitwuzla_set_abort_callback(error_handler);
}

bitwuzla_convt::~bitwuzla_convt()
{
  bitwuzla_delete(bitw);
  bitw = nullptr;
}

smt_convt::resultt bitwuzla_convt::dec_solve()
{
  pre_solve();

  BitwuzlaResult result = bitwuzla_check_sat(bitw);

  if(result == BITWUZLA_SAT)
    return P_SATISFIABLE;

  if(result == BITWUZLA_UNSAT)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string bitwuzla_convt::solver_text()
{
  std::string ss = "Bitwuzla ";
  ss += bitwuzla_version(bitw);
  return ss;
}

void bitwuzla_convt::assert_ast(smt_astt a)
{
  bitwuzla_assert(bitw, to_solver_smt_ast<bitw_smt_ast>(a)->a);
}

smt_astt bitwuzla_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_ADD,  
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SUB,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_MUL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      //BITWUZLA_KIND_BV_SMOD,
      BITWUZLA_KIND_BV_SREM,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_UREM,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SDIV,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_UDIV,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SHL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_ASHR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SHR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    bitwuzla_mk_term1(bitw, BITWUZLA_KIND_BV_NEG, to_solver_smt_ast<bitw_smt_ast>(a)->a), a->sort);
}

smt_astt bitwuzla_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    bitwuzla_mk_term1(bitw, BITWUZLA_KIND_BV_NOT, to_solver_smt_ast<bitw_smt_ast>(a)->a), a->sort);
}

smt_astt bitwuzla_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_XNOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_NOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_NAND,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_XOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_OR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_AND,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_IMPLIES,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_XOR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_OR,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_AND,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    bitwuzla_mk_term1(bitw, BITWUZLA_KIND_NOT, to_solver_smt_ast<bitw_smt_ast>(a)->a), boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_ULT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SLT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_UGT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SGT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_ULE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SLE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_UGE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_SGE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_EQUAL,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_neq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_DISTINCT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt bitwuzla_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term3(
      bitw,
      BITWUZLA_KIND_ARRAY_STORE,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a,
      to_solver_smt_ast<bitw_smt_ast>(c)->a),
    a->sort);
}

smt_astt bitwuzla_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_ARRAY_SELECT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    a->sort->get_range_sort());
}

smt_astt bitwuzla_convt::mk_smt_int(const BigInt &theint [[gnu::unused]])
{
  std::cerr << "Boolector can't create integer sorts" << std::endl;
  abort();
}

smt_astt bitwuzla_convt::mk_smt_real(const std::string &str [[gnu::unused]])
{
  std::cerr << "Boolector can't create Real sorts" << std::endl;
  abort();
}
/*
smt_astt bitwuzla_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  return new_ast(
    boolector_const(btor, integer2binary(theint, s->get_data_width()).c_str()),
    s);
}
*/
smt_astt bitwuzla_convt::mk_smt_bool(bool val)
{
  BitwuzlaTerm *node = (val) ? bitwuzla_mk_true(bitw) : bitwuzla_mk_false(bitw);
  const smt_sort *sort = boolean_sort;
  return new_ast(node, sort);
}

smt_astt bitwuzla_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype [[gnu::unused]])
{
  return mk_smt_symbol(name, s);
}

smt_astt bitwuzla_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  symtable_type::iterator it = symtable.find(name);
  if(it != symtable.end())
    return it->second;

  BitwuzlaTerm *node;

  switch(s->id)
  {
  case SMT_SORT_BV:
  case SMT_SORT_FIXEDBV:
  case SMT_SORT_BVFP:
  case SMT_SORT_BVFP_RM:
    node = bitwuzla_mk_var(
      bitw, to_solver_smt_sort<BitwuzlaSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_BOOL:
    node = bitwuzla_mk_var(
      bitw, to_solver_smt_sort<BitwuzlaSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_ARRAY:
    node = bitwuzla_mk_const_array(
      bitw, to_solver_smt_sort<BitwuzlaSort>(s)->s, name.c_str());
    break;

  default:
    std::cerr << "Unknown type for symbol\n";
    abort();
  }

  smt_astt ast = new_ast(node, s);

  symtable.insert(symtable_type::value_type(name, ast));
  return ast;
}

/*
smt_astt bitwuzla_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  BoolectorNode *b = boolector_slice(btor, ast->a, high, low);
  return new_ast(b, s);
}
*/

smt_astt bitwuzla_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  //BoolectorNode *b = boolector_sext(btor, ast->a, topwidth);
  BitwuzlaTerm *b = bitwuzla_mk_term1_indexed1(bitw, BITWUZLA_KIND_BV_SIGN_EXTEND, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
  //BoolectorNode *b = boolector_uext(btor, ast->a, topwidth);
  BitwuzlaTerm *b = bitwuzla_mk_term1_indexed1(bitw, BITWUZLA_KIND_BV_ZERO_EXTEND, ast->a, topwidth);
  return new_ast(b, s);
}

smt_astt bitwuzla_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    bitwuzla_mk_term2(
      bitw,
      BITWUZLA_KIND_BV_CONCAT,
      to_solver_smt_ast<bitw_smt_ast>(a)->a,
      to_solver_smt_ast<bitw_smt_ast>(b)->a),
    s);
}

smt_astt bitwuzla_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  return new_ast(
    bitwuzla_mk_term3(
      bitw,
      BITWUZLA_KIND_ITE,
      to_solver_smt_ast<bitw_smt_ast>(cond)->a,
      to_solver_smt_ast<bitw_smt_ast>(t)->a,
      to_solver_smt_ast<bitw_smt_ast>(f)->a),
    t->sort);
}
/*
bool bitwuzla_convt::get_bool(smt_astt a)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);
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
    std::cerr << "Can't get boolean value from Boolector\n";
    abort();
  }

  boolector_free_bv_assignment(btor, result);
  return res;
}

BigInt bitwuzla_convt::get_bv(smt_astt a, bool is_signed)
{
  const bitw_smt_ast *ast = to_solver_smt_ast<bitw_smt_ast>(a);

  const char *result = boolector_bv_assignment(btor, ast->a);
  BigInt val = binary2integer(result, is_signed);
  boolector_free_bv_assignment(btor, result);

  return val;
}

expr2tc bitwuzla_convt::get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  // We do the cast directly here instead of using to_solver_smt_ast because
  // in release mode the dynamic_cast is converted to a static_cast, but we
  // want to catch if the conversion fails
  const bitw_smt_ast *ast = dynamic_cast<const bitw_smt_ast *>(array);
  if(ast == nullptr)
    throw new type2t::symbolic_type_excp();

  uint32_t size;
  char **indicies, **values;
  boolector_array_assignment(btor, ast->a, &indicies, &values, &size);

  BigInt val = 0;
  if(size > 0)
  {
    for(uint32_t i = 0; i < size; i++)
    {
      auto idx = string2integer(indicies[i], 2);
      if(idx == index)
      {
        val = binary2integer(values[i], is_signedbv_type(subtype));
        break;
      }
    }

    boolector_free_array_assignment(btor, indicies, values, size);
    return get_by_value(subtype, val);
  }

  return gen_zero(subtype);
}
*/
smt_astt bitwuzla_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);

  const bitw_smt_ast *side1 =
    to_solver_smt_ast<bitw_smt_ast>(convert_ast(opers.side_1));
  const bitw_smt_ast *side2 =
    to_solver_smt_ast<bitw_smt_ast>(convert_ast(opers.side_2));

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed =
    (is_signedbv_type(opers.side_1) || is_signedbv_type(opers.side_2));

  BitwuzlaTerm *res;
  if(is_add2t(overflow.operand))
  {
    if(is_signed)
    {
      //res = boolector_saddo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_SADD_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      //res = boolector_uaddo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_UADD_OVERFLOW, side1->a, side2->a);
    }
  }
  else if(is_sub2t(overflow.operand))
  {
    if(is_signed)
    {
      //res = boolector_ssubo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_SSUB_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      //res = boolector_usubo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_USUB_OVERFLOW, side1->a, side2->a);
    }
  }
  else if(is_mul2t(overflow.operand))
  {
    if(is_signed)
    {
      //res = boolector_smulo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_SMUL_OVERFLOW, side1->a, side2->a);
    }
    else
    {
      //res = boolector_umulo(btor, side1->a, side2->a);
      res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_UMUL_OVERFLOW, side1->a, side2->a);
    }
  }
  else if(is_div2t(overflow.operand) || is_modulus2t(overflow.operand))
  {
    //res = boolector_sdivo(btor, side1->a, side2->a);
    res = bitwuzla_mk_term2(bitw, BITWUZLA_KIND_BV_SDIV_OVERFLOW, side1->a, side2->a);
  }
  else
  {
    return smt_convt::overflow_arith(expr);
  }

  const smt_sort *s = boolean_sort;
  return new_ast(res, s);
}

/*
smt_astt
bitwuzla_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  smt_sortt dom_sort = mk_int_bv_sort(domain_width);
  smt_sortt arrsort = mk_array_sort(dom_sort, init_val->sort);

  return new_ast(
    bitwuzla_mk_const_array(
      bitw,
      to_solver_smt_sort<BitwuzlaSort>(arrsort)->s,
      to_solver_smt_ast<bitw_smt_ast>(init_val)->a),
    arrsort);
}
*/
void bitwuzla_convt::dump_smt()
{
  bitwuzla_dump_formula(bitw, const_cast<char *>("smt2"), stdout);
}

void bitw_smt_ast::dump() const
{
  bitwuzla_term_dump(a, const_cast<char *>("smt2"), stdout);
  std::cout << std::flush;
}

void bitwuzla_convt::print_model()
{
  bitwuzla_print_model(bitw, const_cast<char *>("smt2"), stdout);
}
/*
smt_sortt bitwuzla_convt::mk_bool_sort()
{
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_BOOL, bitwuzla_mk_bool_sort(bitw), 1);
}

smt_sortt bitwuzla_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_BV, bitwuzla_mk_bv_sort(bitw, width), width);
}

smt_sortt bitwuzla_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_FIXEDBV, bitwuzla_mk_bv_sort(bitw, width), width);
}

smt_sortt bitwuzla_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<BitwuzlaSort>(domain);
  auto range_sort = to_solver_smt_sort<BitwuzlaSort>(range);

  auto t = bitwuzla_mk_array_sort(bitw, domain_sort->s, range_sort->s);
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_ARRAY, t, domain_sort->get_data_width(), range);
}

smt_sortt bitwuzla_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_BVFP,
    bitwuzla_mk_bv_sort(bitw, ew + sw + 1),
    ew + sw + 1,
    sw + 1);
}

smt_sortt bitwuzla_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<BitwuzlaSort>(
    SMT_SORT_BVFP_RM, bitwuzla_mk_bv_sort(bitw, 3), 3);
}
*/
