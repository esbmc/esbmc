#include <stp_conv.h>
#include <cstring>

smt_convt *create_new_stp_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api __attribute__((unused)),
  array_iface **array_api,
  fp_convt **fp_api)
{
  stp_convt *conv = new stp_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

void errorHandler(const char *err_msg)
{
  std::cerr << err_msg << '\n';
}

stp_convt::stp_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns), array_iface(false, false), fp_convt(this)
{
  if(int_encoding)
  {
    std::cerr << "STP does not support integer encoding mode" << std::endl;
    abort();
  }

  context = vc_createValidityChecker();
  vc_registerErrorHandler(errorHandler);
}

stp_convt::~stp_convt()
{
}

smt_convt::resultt stp_convt::dec_solve()
{
  pre_solve();

  int result = vc_query(context, vc_falseExpr(context));
  if(result == 0)
    return P_SATISFIABLE;

  if(result == 1)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string stp_convt::solver_text()
{
  return "STP";
}

void stp_convt::assert_ast(const smt_ast *a)
{
  vc_assertFormula(context, to_solver_smt_ast<stp_smt_ast>(a)->a);
}

smt_astt stp_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  return new_ast(
    vc_bvPlusExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvMinusExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvMultExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvModExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvModExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvDivExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvDivExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvLeftShiftExprExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvSignedRightShiftExprExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  return new_ast(
    vc_bvLeftShiftExprExpr(
      context,
      a->sort->get_data_width(),
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    vc_bvUMinusExpr(context, to_solver_smt_ast<stp_smt_ast>(a)->a), a->sort);
}

smt_astt stp_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  return new_ast(
    vc_bvNotExpr(context, to_solver_smt_ast<stp_smt_ast>(a)->a), a->sort);
}

smt_astt stp_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvXorExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvOrExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvAndExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort);
}

smt_astt stp_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    vc_impliesExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    vc_xorExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    vc_orExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    vc_andExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    vc_notExpr(context, to_solver_smt_ast<stp_smt_ast>(a)->a), boolean_sort);
}

smt_astt stp_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvLtExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvLtExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvGtExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvGtExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvLeExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvLeExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_bvGeExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return new_ast(
    vc_sbvGeExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  if(a->sort->id == SMT_SORT_BOOL || a->sort->id == SMT_SORT_STRUCT)
    return new_ast(
      vc_iffExpr(
        context,
        to_solver_smt_ast<stp_smt_ast>(a)->a,
        to_solver_smt_ast<stp_smt_ast>(b)->a),
      boolean_sort);

  return new_ast(
    vc_eqExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    boolean_sort);
}

smt_astt stp_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return new_ast(
    vc_writeExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a,
      to_solver_smt_ast<stp_smt_ast>(c)->a),
    a->sort);
}

smt_astt stp_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return new_ast(
    vc_readExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    a->sort->get_range_sort());
}

smt_ast *stp_convt::mk_smt_int(
  const mp_integer &theint __attribute__((unused)),
  bool sign __attribute__((unused)))
{
  std::cerr << "STP can't create integer sorts" << std::endl;
  abort();
}

smt_ast *stp_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  std::cerr << "STP can't create Real sorts" << std::endl;
  abort();
}

smt_astt stp_convt::mk_smt_bv(const mp_integer &theint, smt_sortt s)
{
  std::string str = integer2binary(theint, s->get_data_width());
  return new_ast(vc_bvConstExprFromStr(context, str.c_str()), s);
}

smt_ast *stp_convt::mk_smt_bool(bool val)
{
  Expr node = (val) ? vc_trueExpr(context) : vc_falseExpr(context);
  const smt_sort *sort = boolean_sort;
  return new_ast(node, sort);
}

smt_ast *stp_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *stp_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  std::string new_name = name;
  std::replace(new_name.begin(), new_name.end(), '@', '_');
  std::replace(new_name.begin(), new_name.end(), '!', '_');
  std::replace(new_name.begin(), new_name.end(), '&', '_');
  std::replace(new_name.begin(), new_name.end(), '#', '_');
  std::replace(new_name.begin(), new_name.end(), '$', '_');
  std::replace(new_name.begin(), new_name.end(), ':', '_');

  return new_ast(
    vc_varExpr(context, new_name.c_str(), to_solver_smt_sort<Type>(s)->s), s);
}

smt_astt
stp_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  const stp_smt_ast *ast = to_solver_smt_ast<stp_smt_ast>(a);
  return new_ast(vc_bvExtract(context, ast->a, high, low), s);
}

smt_astt stp_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  smt_sortt s = mk_bv_sort(a->sort->get_data_width() + topwidth);
  const stp_smt_ast *ast = to_solver_smt_ast<stp_smt_ast>(a);
  return new_ast(vc_bvSignExtend(context, ast->a, topwidth), s);
}

smt_astt stp_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  smt_astt z = mk_smt_bv(0, mk_bv_sort(topwidth));
  return mk_concat(z, a);
}

smt_astt stp_convt::mk_concat(smt_astt a, smt_astt b)
{
  smt_sortt s =
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width());

  return new_ast(
    vc_bvConcatExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(a)->a,
      to_solver_smt_ast<stp_smt_ast>(b)->a),
    s);
}

smt_astt stp_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  return new_ast(
    vc_iteExpr(
      context,
      to_solver_smt_ast<stp_smt_ast>(cond)->a,
      to_solver_smt_ast<stp_smt_ast>(t)->a,
      to_solver_smt_ast<stp_smt_ast>(f)->a),
    t->sort);
}

bool stp_convt::get_bool(const smt_ast *a)
{
  Expr expr_value =
    vc_getCounterExample(context, to_solver_smt_ast<stp_smt_ast>(a)->a);
  int res = getBVUnsigned(vc_boolToBVExpr(context, expr_value));
  return res != 0;
}

BigInt stp_convt::get_bv(smt_astt a)
{
  Expr expr_value =
    vc_getCounterExample(context, to_solver_smt_ast<stp_smt_ast>(a)->a);
  return getBVUnsigned(expr_value);
}

expr2tc stp_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  const stp_smt_ast *ast = to_solver_smt_ast<stp_smt_ast>(array);

  Expr *indices;
  Expr *values;
  int size;
  vc_getCounterExampleArray(context, ast->a, &indices, &values, &size);

  return get_by_ast(
    subtype,
    new_ast(
      vc_getCounterExample(context, values[index]), convert_sort(subtype)));
}

const smt_ast *
stp_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void stp_convt::add_array_constraints_for_solving()
{
}

void stp_convt::push_array_ctx()
{
}

void stp_convt::pop_array_ctx()
{
}

void stp_convt::dump_smt()
{
  vc_printQuery(context);
}

void stp_smt_ast::dump() const
{
  auto convt = dynamic_cast<const stp_convt *>(context);
  assert(convt != nullptr);

  vc_printExpr(convt->context, a);
}

void stp_convt::print_model()
{
  vc_printCounterExample(context);
}

smt_sortt stp_convt::mk_bool_sort()
{
  return new solver_smt_sort<Type>(SMT_SORT_BOOL, vc_boolType(context), 1);
}

smt_sortt stp_convt::mk_bv_sort(std::size_t width)
{
  return new solver_smt_sort<Type>(
    SMT_SORT_BV, vc_bvType(context, width), width);
}

smt_sortt stp_convt::mk_fbv_sort(std::size_t width)
{
  return new solver_smt_sort<Type>(
    SMT_SORT_FIXEDBV, vc_bvType(context, width), width);
}

smt_sortt stp_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<Type>(domain);
  auto range_sort = to_solver_smt_sort<Type>(range);

  auto t = vc_arrayType(context, domain_sort->s, range_sort->s);
  return new solver_smt_sort<Type>(
    SMT_SORT_ARRAY, t, domain_sort->get_data_width(), range);
}

smt_sortt stp_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<Type>(
    SMT_SORT_BVFP, vc_bvType(context, ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt stp_convt::mk_bvfp_rm_sort()
{
  return new solver_smt_sort<Type>(SMT_SORT_BVFP_RM, vc_bvType(context, 3), 3);
}
