/*******************************************************************
   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <assert.h>
#include <ctype.h>
#include <malloc.h>
#include <stdarg.h>
#include <fstream>
#include <sstream>
#include <std_expr.h>
#include <irep2.h>
#include <migrate.h>
#include <arith_tools.h>
#include <std_types.h>
#include <config.h>
#include <i2string.h>
#include <expr_util.h>
#include <string2array.h>
#include <pointer_offset_size.h>
#include <find_symbols.h>
#include <prefix.h>
#include <fixedbv.h>
#include <base_type.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

#define cast_to_z3(arg) (*(reinterpret_cast<z3::expr *&>((arg))))
#define cast_to_z3_sort(arg) (*(reinterpret_cast<z3::sort *>((arg))))

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                          "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

Z3_ast workaround_Z3_mk_bvadd_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvadd_no_underflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_overflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvneg_no_overflow(Z3_context ctx, Z3_ast a);
z3_convt::z3_convt(bool int_encoding, bool is_cpp, const namespacet &_ns)
: smt_convt(caching, int_encoding, _ns, is_cpp)
{
  this->int_encoding = int_encoding;

  assumpt_mode = false;
  no_variables = 1;

  z3::config conf;
  conf.set("MODEL", true);
  conf.set("RELEVANCY", 0);
  conf.set("SOLVER", true);
  // Disabling this option results in the enablement of --symex-thread-guard on
  // 03_exor_01 to not explode solving time. No idea why this is the case,
  // doesn't affect any other solving time.
  conf.set("ARRAY_ALWAYS_PROP_UPWARD", false);

  ctx.init(conf, int_encoding);

  z3_ctx = ctx;
  Z3_set_ast_print_mode(z3_ctx, Z3_PRINT_SMTLIB_COMPLIANT);

  solver = z3::solver(ctx);

  setup_pointer_sort();
  total_mem_space.push_back(0);

  assumpt_ctx_stack.push_back(assumpt.begin());

  smt_convt::init_addr_space_array();
  z3_convt::init_addr_space_array();
}


z3_convt::~z3_convt()
{

  // jmorse - remove when smtlib printer exists and works.
#if 0
  if (smtlib) {
    std::ofstream temp_out;
    Z3_string smt_lib_str, logic;
    Z3_ast assumpt_array_ast[assumpt.size() + 1];
    z3::expr formula;
    formula = ctx.bool_val(true);

    std::list<z3::expr>::const_iterator it;
    unsigned int i;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumpt_array_ast[i] = *it;
    }

    if (int_encoding)
      logic = "QF_AUFLIRA";
    else
      logic = "QF_AUFBV";

    smt_lib_str = Z3_benchmark_to_smtlib_string(z3_ctx, "ESBMC", logic,
                                    "unknown", "", assumpt.size(),
                                    assumpt_array_ast, formula);

    temp_out.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);

    temp_out << smt_lib_str << std::endl;
  }
#endif
}

void
z3_convt::push_ctx(void)
{

  smt_convt::push_ctx();
  intr_push_ctx();
  solver.push();
}

void
z3_convt::pop_ctx(void)
{

  solver.pop();
  intr_pop_ctx();
  smt_convt::pop_ctx();;

  // Clear model if we have one.
  model = z3::model();
}

void
z3_convt::intr_push_ctx(void)
{

  // Also push/duplicate pointer logic state.
  total_mem_space.push_back(total_mem_space.back());

  // Store where we are in the list of assumpts.
  std::list<z3::expr>::iterator it = assumpt.end();
  it--;
  assumpt_ctx_stack.push_back(it);
}

void
z3_convt::intr_pop_ctx(void)
{

  // Erase everything on stack since last push_ctx
  std::list<z3::expr>::iterator it = assumpt_ctx_stack.back();
  ++it;
  assumpt.erase(it, assumpt.end());
  assumpt_ctx_stack.pop_back();

  total_mem_space.pop_back();
}

void
z3_convt::init_addr_space_array(void)
{
  z3::symbol mk_tuple_name, proj_names[2];
  Z3_symbol proj_names_sym[2];
  Z3_sort proj_types[2];
  Z3_func_decl mk_tuple_decl, proj_decls[2];

  z3::sort tmp_proj_type = ctx.esbmc_int_sort();
  proj_types[0] = proj_types[1] = tmp_proj_type;

  mk_tuple_name = z3::symbol(ctx, "struct_type_addr_space_tuple");
  proj_names[0] = z3::symbol(ctx, "start");
  proj_names[1] = z3::symbol(ctx, "end");
  proj_names_sym[0] = proj_names[0];
  proj_names_sym[1] = proj_names[1];

  addr_space_tuple_sort = z3::to_sort(ctx, Z3_mk_tuple_sort(
                                      ctx, mk_tuple_name, 2,
                                      proj_names_sym, proj_types,
                                      &mk_tuple_decl, proj_decls));
  Z3_func_decl tmp_addr_space_decl =
    Z3_get_tuple_sort_mk_decl(ctx, addr_space_tuple_sort);
  addr_space_tuple_decl = z3::func_decl(ctx, tmp_addr_space_decl);

  addr_space_arr_sort = 
                  ctx.array_sort(ctx.esbmc_int_sort(), addr_space_tuple_sort);

  return;
}

prop_convt::resultt
z3_convt::dec_solve(void)
{
  unsigned major, minor, build, revision;
  z3::check_result result;
  Z3_get_version(&major, &minor, &build, &revision);

  std::cout << "Solving with SMT Solver Z3 v" << major << "." << minor << "\n";

  result = check2_z3_properties();

  if (result == z3::unsat)
    return prop_convt::P_UNSATISFIABLE;
  else if (result == z3::unknown)
    return prop_convt::P_ERROR;
  else
    return prop_convt::P_SATISFIABLE;
}

z3::check_result
z3_convt::check2_z3_properties(void)
{
  z3::check_result result;
  unsigned i;
  std::string literal;
  z3::expr_vector assumptions(ctx);

  if (assumpt_mode) {
    std::list<z3::expr>::const_iterator it;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumptions.push_back(*it);
    }
  }

  // XXX XXX XXX jmorse: as of 5dd8a432 running with --smt-during-symex on tests
  // like 03_exor_01 caused a significant performance hit for no known reason.
  // Solving got progressively slower as more interleavings were checked.
  // Profiling said a lot of time was spent in Z3's
  // bv_simplifier_plugin::bit2bool_simplify method. This doesn't happen if you
  // run with no additional options. No idea why, but the belief is that the
  // solver is caching something, bloats, and leads to a performance hit.
  //
  // So during debugging I added the following line to see whether some asserts
  // were being left in the solver accidentally leading to the bloat and... it
  // just stopped. Presumably this accidentally flushes some kind of internal
  // cache and kills bloatage; I've no idea why; but if you remove it there's
  // a significant performance hit.
  z3::expr_vector vec = solver.assertions();

  if (assumpt_mode) {
    result = solver.check(assumptions);
  } else {
    result = solver.check();
  }

  if (result == z3::sat)
    model = solver.get_model();

  if (config.options.get_bool_option("dump-z3-assigns") && result == z3::sat)
    std::cout << Z3_model_to_string(z3_ctx, model);

  return result;
}

void
z3_convt::convert_struct_union_type(const std::vector<type2tc> &members,
                                    const std::vector<irep_idt> &member_names,
                                    const irep_idt &struct_name, bool uni,
                                    void *_bv)
{
  z3::symbol mk_tuple_name, *proj_names;
  z3::sort *proj_types;
  z3::sort &sort = cast_to_z3_sort(_bv);
  Z3_func_decl mk_tuple_decl, *proj_decls;
  std::string name;
  u_int num_elems;

  num_elems = members.size();
  if (uni)
    num_elems++;

  proj_names = new z3::symbol[num_elems];
  proj_types = new z3::sort[num_elems];
  proj_decls = new Z3_func_decl[num_elems];

  name = ((uni) ? "union" : "struct" );
  name += "_type_" + struct_name.as_string();
  mk_tuple_name = z3::symbol(ctx, name.c_str());

  if (!members.size()) {
    sort = z3::to_sort(ctx, Z3_mk_tuple_sort(ctx, mk_tuple_name, 0, NULL, NULL, &mk_tuple_decl, NULL));
    return;
  }

  u_int i = 0;
  std::vector<irep_idt>::const_iterator mname = member_names.begin();
  for (std::vector<type2tc>::const_iterator it = members.begin();
       it != members.end(); it++, mname++, i++)
  {
    proj_names[i] = z3::symbol(ctx, mname->as_string().c_str());
    convert_type(*it, proj_types[i]);
  }

  if (uni) {
    // ID field records last value written to union
    proj_names[num_elems - 1] = z3::symbol(ctx, "id");
    // XXXjmorse - must this field really become a bitfield, ever? It's internal
    // tracking data, not program data.
    proj_types[num_elems - 1] = ctx.esbmc_int_sort();
  }

  // Unpack pointers from Z3++ objects.
  Z3_symbol *unpacked_symbols = new Z3_symbol[num_elems];
  Z3_sort *unpacked_sorts = new Z3_sort[num_elems];
  for (i = 0; i < num_elems; i++) {
    unpacked_symbols[i] = proj_names[i];
    unpacked_sorts[i] = proj_types[i];
  }

  sort = z3::to_sort(ctx, Z3_mk_tuple_sort(ctx, mk_tuple_name, num_elems,
                           unpacked_symbols, unpacked_sorts, &mk_tuple_decl,
                           proj_decls));

  delete[] unpacked_symbols;
  delete[] unpacked_sorts;
  delete[] proj_names;
  delete[] proj_types;
  delete[] proj_decls;

  return;
}

void
z3_convt::setup_pointer_sort(void)
{
  z3::sort s;
  convert_type(pointer_struct, s);
  pointer_sort = s;
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(ctx, s);
  pointer_decl = z3::func_decl(ctx, decl);
  return;
}

void
z3_convt::convert_struct_union(const std::vector<expr2tc> &members,
                               const std::vector<type2tc> &member_types,
                               const type2tc &type, z3::expr &output)
{

  // Converts a static struct/union - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  u_int i = 0;

  z3::sort sort;
  convert_type(type, sort);

  unsigned size = member_types.size();

  z3::expr *args = new z3::expr[size];

  unsigned int numoperands = members.size();
  // Populate tuple with members of that struct/union
  forall_types(it, member_types) {
    if (i < numoperands) {
      const z3_smt_ast *tmp = z3_smt_downcast(convert_ast(members[i]));
      args[i] = tmp->e;
    } else {
      // Turns out that unions don't necessarily initialize all members.
      // If no initialization give, use free (fresh) variable.
      z3::sort s;
      convert_type(*it, s);
      args[i] = ctx.fresh_const(NULL, s);
    }

    i++;
  }

  // Create tuple itself, return to caller. This is a lump of data, we don't
  // need to bind it to a name or symbol.
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(ctx, sort);
  z3::func_decl d(ctx, decl);
  output = d.make_tuple_from_array(size, args);
  delete[] args;
}

void
z3_convt::convert_type(const type2tc &type, z3::sort &sort)
{

  switch (type->type_id) {
  case type2t::bool_id:
    sort = ctx.bool_sort();
    break;
  case type2t::struct_id:
  {
    const struct_type2t &strct = to_struct_type(type);
    convert_struct_union_type(strct.members, strct.member_names, strct.name,
                              false, &sort);
    break;
  }
  case type2t::union_id:
  {
    const union_type2t &uni = to_union_type(type);
    convert_struct_union_type(uni.members, uni.member_names, uni.name,
                              true, &sort);
    break;
  }
  case type2t::array_id:
  {
    const array_type2t &arr = to_array_type(type);
    z3::sort subtype;
    convert_type(arr.subtype, subtype);
    sort = ctx.array_sort(ctx.esbmc_int_sort(), subtype);
    break;
  }
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  {
    if (int_encoding) {
      sort = ctx.esbmc_int_sort();
    } else {
      unsigned int width = type->get_width();
      sort = ctx.bv_sort(width);
    }
    break;
  }
  case type2t::fixedbv_id:
  {
    unsigned int width = type->get_width();

    if (int_encoding)
      sort = ctx.real_sort();
    else
      sort = ctx.bv_sort(width);
    break;
  }
  case type2t::pointer_id:
    convert_type(pointer_struct, sort);
    break;
  case type2t::string_id:
  case type2t::code_id:
  default:
    std::cerr << "Invalid type ID being converted to Z3 sort" << std::endl;
    type->dump();
    abort();
  }

  return;
}

literalt
z3_convt::land(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  z3::expr args[size];
  Z3_ast args_ast[size];
  z3::expr result, formula;

  for (unsigned int i = 0; i < bv.size(); i++) {
    args[i] = z3_literal(bv[i]);
    args_ast[i] = args[i];
  }

  result = to_expr(ctx, Z3_mk_and(z3_ctx, bv.size(), args_ast));
  z3::expr thelit = z3_literal(l);
  formula = to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
  assert_formula(formula);

  return l;
}

literalt
z3_convt::lor(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  z3::expr args[size];
  Z3_ast args_ast[size];
  z3::expr result, formula;

  for (unsigned int i = 0; i < bv.size(); i++) {
    args[i] = z3_literal(bv[i]);
    args_ast[i] = args[i];
  }

  result = z3::to_expr(ctx, Z3_mk_or(z3_ctx, bv.size(), args_ast));

  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
  assert_formula(formula);

  return l;
}

literalt
z3_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  literalt l = new_variable();
  z3::expr result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = operand[0] && operand[1];
  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
  assert_formula(formula);

  return l;
}

literalt
z3_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);
  if (a == b) return a;

  literalt l = new_variable();
  z3::expr result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = operand[0] || operand[1];
  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
  assert_formula(formula);

  return l;
}

literalt
z3_convt::lnot(literalt a)
{
  a.invert();

  return a;
}

literalt
z3_convt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
z3_convt::new_variable()
{
  literalt l;

  l.set(no_variables, false);

  set_no_variables(no_variables + 1);

  return l;
}

bool
z3_convt::process_clause(const bvt &bv, bvt &dest)
{

  dest.clear();

  // empty clause! this is UNSAT
  if (bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin();
       it != bv.end();
       it++)
  {
    literalt l = *it;

    // we never use index 0
    assert(l.var_no() != 0);

    if (l.is_true())
      return true;  // clause satisfied

    if (l.is_false())
      continue;

    assert(l.var_no() < no_variables);

    // prevent duplicate literals
    if (s.insert(l).second)
      dest.push_back(l);

    if (s.find(lnot(l)) != s.end())
      return true;  // clause satisfied
  }

  return false;
}

void
z3_convt::lcnf(const bvt &bv)
{

  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.size() == 0)
    return;

  z3::expr lor_var, args[new_bv.size()];
  Z3_ast args_ast[new_bv.size()];
  unsigned int i = 0;

  for (bvt::const_iterator it = new_bv.begin(); it != new_bv.end(); it++, i++) {
    args[i] = z3_literal(*it);
    args_ast[i] = args[i];
  }

  if (i > 1) {
    lor_var = z3::expr(ctx, Z3_mk_or(z3_ctx, i, args_ast));
    assert_formula(lor_var);
  } else   {
    assert_formula(args[0]);
  }
}

z3::expr
z3_convt::z3_literal(literalt l)
{

  z3::expr literal_l;
  std::string literal_s;

  if (l == const_literal(false))
    return ctx.bool_val(false);
  else if (l == const_literal(true))
    return ctx.bool_val(true);

  literal_s = "l" + i2string(l.var_no());
  literal_l = ctx.constant(literal_s.c_str(), ctx.bool_sort());

  if (l.sign()) {
    return !literal_l;
  }

  return literal_l;
}

tvt
z3_convt::l_get(literalt a)
{
  tvt result = tvt(tvt::TV_ASSUME);
  std::string literal;

  if (a.is_true()) {
    return tvt(true);
  } else if (a.is_false())    {
    return tvt(false);
  }

  symbol2tc sym(get_bool_type(), irep_idt("l" + i2string(a.var_no())));
  expr2tc res = get(sym);

  if (!is_nil_expr(res) && is_constant_bool2t(res)) {
    result = (to_constant_bool2t(res).is_true())
             ? tvt(tvt::TV_TRUE) : tvt(tvt::TV_FALSE);
  } else {
    result = tvt(tvt::TV_UNKNOWN);
  }

  if (a.sign())
    result = !result;

  return result;
}

void
z3_convt::assert_lit(const literalt &l)
{
  z3::expr thelit = z3_literal(l);
  solver.add(thelit);
  assumpt.push_back(thelit);
}

void
z3_convt::assert_formula(const z3::expr &ast)
{

  // If we're not going to be using the assumptions (ie, for unwidening and for
  // smtlib) then just assert the fact to be true.
  if (!assumpt_mode) {
    solver.add(ast);
    return;
  }

  literalt l = new_variable();
  z3::expr thelit = z3_literal(l);
  z3::expr formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, ast));
  solver.add(formula);

  // jmorse - delete when smtlib printer exists and works
#if 0
  if (smtlib)
    assumpt.push_back(ast);
  else
#endif
    assumpt.push_back(z3_literal(l));

  return;
}

z3::expr
z3_convt::mk_tuple_update(const z3::expr &t, unsigned i, const z3::expr &newval)
{
  z3::sort ty;
  unsigned num_fields, j;

  ty = t.get_sort();

  if (!ty.is_datatype()) {
    std::cerr << "argument must be a tuple";
    abort();
  }

  num_fields = Z3_get_tuple_sort_num_fields(ctx, ty);

  if (i >= num_fields) {
    std::cerr << "invalid tuple update, index is too big";
    abort();
  }

  z3::expr new_fields[num_fields];
  for (j = 0; j < num_fields; j++) {
    if (i == j) {
      /* use new_val at position i */
      new_fields[j] = newval;
    } else   {
      /* use field j of t */
      z3::func_decl proj_decl =
        z3::to_func_decl(ctx, Z3_get_tuple_sort_field_decl(ctx, ty, j));
      new_fields[j] = proj_decl(t);
    }
  }

  z3::func_decl mk_tuple_decl =
    z3::to_func_decl(ctx, Z3_get_tuple_sort_mk_decl(ctx, ty));

  return mk_tuple_decl.make_tuple_from_array(num_fields, new_fields);
}

z3::expr
z3_convt::mk_tuple_select(const z3::expr &t, unsigned i)
{
  z3::sort ty;
  unsigned num_fields;

  ty = t.get_sort();

  if (!ty.is_datatype()) {
    throw new z3_convt::conv_error("argument must be a tuple");
  }

  num_fields = Z3_get_tuple_sort_num_fields(ctx, ty);

  if (i >= num_fields) {
    throw new z3_convt::conv_error("invalid tuple select, index is too big");
  }

  z3::func_decl proj_decl =
    z3::to_func_decl(ctx, Z3_get_tuple_sort_field_decl(ctx, ty, i));
  return proj_decl(t);
}

// SMT-abstraction migration routines.

smt_ast *
z3_convt::mk_func_app(const smt_sort *s, smt_func_kind k, const smt_ast **args, unsigned int numargs, const expr2tc &temp)
{
  const z3_smt_ast *asts[4];
  unsigned int i;

  assert(numargs <= 4);
  for (i = 0; i < numargs; i++)
    asts[i] = z3_smt_downcast(args[i]);

  // So: this method is liable to become one /huge/ switch case that deals with
  // the conversion of most SMT function applications. This normally would
  // be bad; however I figure that if _all_ logic is handled at the higher SMT
  // layer, and all this method does is actually pass arguments through to
  // the solver, then that's absolutely fine.
  switch (k) {
  case SMT_FUNC_ADD:
  case SMT_FUNC_BVADD:
    return new z3_smt_ast(mk_add(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
    return new z3_smt_ast(mk_sub(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
    return new z3_smt_ast((asts[0]->e * asts[1]->e), s, temp);
  case SMT_FUNC_MOD:
    return new z3_smt_ast(
                    z3::to_expr(ctx, Z3_mk_mod(z3_ctx, asts[0]->e, asts[1]->e)),
                    s, temp);
  case SMT_FUNC_BVSMOD:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvsrem(z3_ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_BVUMOD:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvurem(z3_ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_DIV:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_div(z3_ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_BVSDIV:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvsdiv(z3_ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_BVUDIV:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvudiv(z3_ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_SHL:
    return new z3_smt_ast(asts[0]->e * pw(ctx.int_val(2), asts[1]->e), s, temp);
  case SMT_FUNC_BVSHL:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvshl(ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_BVASHR:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvashr(ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_NEG:
  case SMT_FUNC_BVNEG:
    return new z3_smt_ast((-asts[0]->e), s, temp);
  case SMT_FUNC_BVLSHR:
    return new z3_smt_ast(
                 z3::to_expr(ctx, Z3_mk_bvlshr(ctx, asts[0]->e, asts[1]->e)),
                 s, temp);
  case SMT_FUNC_BVNOT:
    return new z3_smt_ast((~asts[0]->e), s, temp);
  case SMT_FUNC_BVNXOR:
    return new z3_smt_ast(mk_bvxnor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_BVNOR:
    return new z3_smt_ast(mk_bvnor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_BVNAND:
    return new z3_smt_ast(mk_bvnor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_BVXOR:
    return new z3_smt_ast(mk_bvxor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_BVOR:
    return new z3_smt_ast(mk_bvor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_BVAND:
    return new z3_smt_ast(mk_bvand(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_IMPLIES:
    return new z3_smt_ast(mk_implies(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_XOR:
    return new z3_smt_ast(mk_xor(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_OR:
    return new z3_smt_ast(mk_or(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_AND:
    return new z3_smt_ast(mk_and(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_NOT:
    return new z3_smt_ast(!asts[0]->e, s, temp);
  // NB: mk_{l,g}t{,e} ignore unsigned arg in integer mode.
  case SMT_FUNC_LT:
  case SMT_FUNC_BVULT:
    return new z3_smt_ast(mk_lt(asts[0]->e, asts[1]->e, true), s, temp);
  case SMT_FUNC_BVSLT:
    return new z3_smt_ast(mk_lt(asts[0]->e, asts[1]->e, false), s, temp);
  case SMT_FUNC_GT:
  case SMT_FUNC_BVUGT:
    return new z3_smt_ast(mk_gt(asts[0]->e, asts[1]->e, true), s, temp);
  case SMT_FUNC_BVSGT:
    return new z3_smt_ast(mk_gt(asts[0]->e, asts[1]->e, false), s, temp);
  case SMT_FUNC_LTE:
  case SMT_FUNC_BVULTE:
    return new z3_smt_ast(mk_le(asts[0]->e, asts[1]->e, true), s, temp);
  case SMT_FUNC_BVSLTE:
    return new z3_smt_ast(mk_le(asts[0]->e, asts[1]->e, false), s, temp);
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVUGTE:
    return new z3_smt_ast(mk_ge(asts[0]->e, asts[1]->e, true), s, temp);
  case SMT_FUNC_BVSGTE:
    return new z3_smt_ast(mk_ge(asts[0]->e, asts[1]->e, false), s, temp);
  case SMT_FUNC_EQ:
    return new z3_smt_ast((asts[0]->e == asts[1]->e), s, temp);
  case SMT_FUNC_NOTEQ:
    return new z3_smt_ast((asts[0]->e != asts[1]->e), s, temp);
  case SMT_FUNC_ITE:
    return new z3_smt_ast(ite(asts[0]->e, asts[1]->e, asts[2]->e), s, temp);
  case SMT_FUNC_STORE:
    return new z3_smt_ast(store(asts[0]->e, asts[1]->e, asts[2]->e), s, temp);
  case SMT_FUNC_SELECT:
    return new z3_smt_ast(select(asts[0]->e, asts[1]->e), s, temp);
  case SMT_FUNC_CONCAT:
    return new z3_smt_ast(
                   z3::to_expr(ctx, Z3_mk_concat(ctx, asts[0]->e, asts[1]->e)),
                   s, temp);
  case SMT_FUNC_REAL2INT:
    return new z3_smt_ast(z3::to_expr(ctx, Z3_mk_real2int(ctx, asts[0]->e)),
                          s, temp);
  case SMT_FUNC_INT2REAL:
    return new z3_smt_ast(z3::to_expr(ctx, Z3_mk_int2real(ctx, asts[0]->e)),
                          s, temp);
  case SMT_FUNC_POW:
        return new z3_smt_ast(z3::to_expr(ctx, Z3_mk_power(ctx, asts[0]->e,                                                       asts[1]->e)),
                                      s, temp);
  case SMT_FUNC_HACKS:
  default:
    assert(0 && "Unhandled SMT func in z3 conversion");
  }
}

smt_ast *
z3_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low,
                     const smt_sort *s, const expr2tc &tmp)
{

  return new z3_smt_ast(z3::to_expr(ctx, Z3_mk_extract(ctx, high, low,
                                         z3_smt_downcast(a)->e)), s, tmp);
}

smt_ast *
z3_convt::mk_smt_int(const mp_integer &theint, bool sign, const expr2tc &temp)
{
  smt_sort *s = mk_sort(SMT_SORT_INT, sign);
  if (theint.is_negative())
    return new z3_smt_ast(ctx.int_val(theint.to_int64()), s, temp);
  else
    return new z3_smt_ast(ctx.int_val(theint.to_uint64()), s, temp);
}

smt_ast *
z3_convt::mk_smt_real(const mp_integer &theval, const expr2tc &temp)
{
  smt_sort *s = mk_sort(SMT_SORT_REAL);
  return new z3_smt_ast(ctx.real_val(theval.to_int64()), s, temp);
}

smt_ast *
z3_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int width, const expr2tc &temp)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, width, sign);
  if (theint.is_negative())
    return new z3_smt_ast(ctx.bv_val(theint.to_int64(), width), s, temp);
  else
    return new z3_smt_ast(ctx.bv_val(theint.to_uint64(), width), s, temp);
}

smt_ast *
z3_convt::mk_smt_bool(bool val, const expr2tc &temp)
{
  smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast(ctx.bool_val(val), s, temp);
}

smt_ast *
z3_convt::mk_smt_symbol(const std::string &name, const smt_sort *s, const expr2tc &temp)
{
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);
  return new z3_smt_ast(ctx.constant(name.c_str(), zs->s), s, temp);
}

smt_sort *
z3_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;
  z3_smt_sort *s = NULL, *dom, *range;
  unsigned long uint;
  int thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    thebool = va_arg(ap, int);
    s = new z3_smt_sort(k, ctx.int_sort(), thebool);
    break;
  case SMT_SORT_REAL:
    s = new z3_smt_sort(k, ctx.real_sort());
    break;
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    s = new z3_smt_sort(k, ctx.bv_sort(uint), thebool);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, z3_smt_sort *); // Consider constness?
    range = va_arg(ap, z3_smt_sort *);
    s = new z3_smt_sort(k, ctx.array_sort(dom->s, range->s));
    break;
  case SMT_SORT_BOOL:
    s = new z3_smt_sort(k, ctx.bool_sort());
    break;
  default:
    assert(0);
  }

  return s;
}

smt_sort *
z3_convt::mk_struct_sort(const type2tc &type)
{
  z3::sort s;
  convert_type(type, s);
  return new z3_smt_sort(SMT_SORT_STRUCT, s);
}

smt_sort *
z3_convt::mk_union_sort(const type2tc &type)
{
  z3::sort s;
  convert_type(type, s);
  return new z3_smt_sort(SMT_SORT_UNION, s);
}

literalt
z3_convt::mk_lit(const smt_ast *a)
{
  const z3_smt_ast *b = static_cast<const z3_smt_ast *>(a);
  literalt l = new_variable();
  z3::expr eq = z3_literal(l) == b->e;
  assert_formula(eq);
  return l;
}

smt_ast *
z3_convt::tuple_create(const expr2tc &structdef)
{
  z3::expr e;
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type =
    static_cast<const struct_union_data &>(*strct.type);

  convert_struct_union(strct.datatype_members, type.members, strct.type, e);
  smt_sort *s = mk_struct_sort(structdef->type);
  return new z3_smt_ast(e, s, structdef);
}

smt_ast *
z3_convt::tuple_project(const smt_ast *a, const smt_sort *s, unsigned int field, const expr2tc &tmp)
{
  const z3_smt_ast *za = z3_smt_downcast(a);
  return new z3_smt_ast(mk_tuple_select(za->e, field), s, tmp);
}

smt_ast *
z3_convt::tuple_update(const smt_ast *a, unsigned int field, const smt_ast *val, const expr2tc &tmp)
{
  const z3_smt_ast *za = z3_smt_downcast(a);
  const z3_smt_ast *zu = z3_smt_downcast(val);
  return new z3_smt_ast(mk_tuple_update(za->e, field, zu->e), za->sort, tmp);
}

smt_ast *
z3_convt::tuple_equality(const smt_ast *a, const smt_ast *b, const expr2tc &tmp)
{
  const z3_smt_ast *za = z3_smt_downcast(a);
  const z3_smt_ast *zb = z3_smt_downcast(b);
  const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast((za->e == zb->e), sort, tmp);
}

smt_ast *
z3_convt::tuple_ite(const smt_ast *cond, const smt_ast *true_val,
                    const smt_ast *false_val, const smt_sort *sort,
                    const expr2tc &tmp)
{

  return new z3_smt_ast(z3::ite(z3_smt_downcast(cond)->e,
                                z3_smt_downcast(true_val)->e,
                                z3_smt_downcast(false_val)->e), sort, tmp);
}

smt_ast *
z3_convt::tuple_array_create(const expr2tc &expr, const smt_sort *domain)
{
  z3::expr output;
  const array_type2t &arrtype = to_array_type(expr->type);

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &array = to_constant_array_of2t(expr);
    z3::expr value, index;
    z3::sort array_type;
    std::string tmp, identifier;
    int64_t size;
    u_int j;

    const constant_int2t &sz = to_constant_int2t(arrtype.array_size);

    convert_type(array.type, array_type);

    if (arrtype.size_is_infinite) {
      // Don't attempt to do anything with this. The user is on their own.
      output = ctx.fresh_const(NULL, array_type);
      goto out;
    }

    assert(is_constant_int2t(arrtype.array_size) &&
           "array_of sizes should be constant");

    size = sz.as_long();

    const z3_smt_ast *tmpast = z3_smt_downcast(convert_ast(array.initializer));
    value = tmpast->e;

    if (is_bool_type(arrtype.subtype)) {
      value = ctx.bool_val(false);
    }

    output = ctx.fresh_const(NULL, array_type);

    //update array
    for (j = 0; j < size; j++)
    {
      index = ctx.esbmc_int_val(j);
      output = z3::store(output, index, value);
    }
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &array = to_constant_array2t(expr);

    u_int i = 0;
    z3::sort z3_array_type;
    z3::expr int_cte, val_cte;
    z3::sort elem_type;

    convert_type(arrtype.subtype, elem_type);
    z3_array_type = ctx.array_sort(ctx.esbmc_int_sort(), elem_type);

    output = ctx.fresh_const(NULL, z3_array_type);

    i = 0;
    forall_exprs(it, array.datatype_members) {
      int_cte = ctx.esbmc_int_val(i);
      const z3_smt_ast *tmpast = z3_smt_downcast(convert_ast(*it));
      output = z3::store(output, int_cte, tmpast->e);
      ++i;
    }
  }

out:
  smt_sort *ssort = mk_struct_sort(arrtype.subtype);
  smt_sort *asort = mk_sort(SMT_SORT_ARRAY, domain, ssort);
  return new z3_smt_ast(output, asort, expr);
}

smt_ast *
z3_convt::tuple_array_select(const smt_ast *a, const smt_sort *s,
                             const smt_ast *idx, const expr2tc &tmp)
{

  z3::expr output = select(z3_smt_downcast(a)->e, z3_smt_downcast(idx)->e);
  return new z3_smt_ast(output, s, tmp);
}

smt_ast *
z3_convt::tuple_array_update(const smt_ast *a, const smt_ast *field,
                             const smt_ast *val, const expr2tc &tmp)
{
  Z3_ast ast = Z3_mk_store(z3_ctx, z3_smt_downcast(a)->e,
                          z3_smt_downcast(field)->e, z3_smt_downcast(val)->e);
  z3::expr output = z3::to_expr(ctx, ast);
  return new z3_smt_ast(output, a->sort, tmp);
}


smt_ast *
z3_convt::tuple_array_equality(const smt_ast *a, const smt_ast *b,
                             const expr2tc &tmp)
{
  z3::expr e = z3_smt_downcast(a)->e == z3_smt_downcast(b)->e;
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast(e, s, tmp);
}

smt_ast *
z3_convt::tuple_array_ite(const smt_ast *cond, const smt_ast *trueval,
                          const smt_ast *false_val, const smt_sort *sort,
                          const expr2tc &expr)
{
  z3::expr output = z3::ite(z3_smt_downcast(cond)->e,
                            z3_smt_downcast(trueval)->e,
                            z3_smt_downcast(false_val)->e);
  return new z3_smt_ast(output, sort, expr);
}

smt_ast *
z3_convt::mk_fresh(const smt_sort *sort)
{
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(sort);
  return new z3_smt_ast(ctx.fresh_const(NULL, zs->s), sort, expr2tc());
}

smt_ast *
z3_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  z3::expr output;
  z3::expr result[2], operand[2];
  unsigned width_op0, width_op1;

  // XXX jmorse - we can't tell whether or not we're supposed to be treating
  // the _result_ as being a signedbv or an unsignedbv, because we only have
  // operands. Ideally, this needs to be encoded somewhere.
  // Specifically, when irep2 conversion reaches code creation, we should
  // encode the resulting type in the overflow operands type. Right now it's
  // inferred.
  Z3_bool is_signed = Z3_L_FALSE;

  typedef Z3_ast (*type1)(Z3_context, Z3_ast, Z3_ast, Z3_bool);
  typedef Z3_ast (*type2)(Z3_context, Z3_ast, Z3_ast);
  type1 call1;
  type2 call2;

  // Unseen downside of flattening templates. Should consider reformatting
  // typecast2t.
  if (is_add2t(overflow.operand)) {
    const smt_ast *o1 = convert_ast(to_add2t(overflow.operand).side_1);
    const smt_ast *o2 = convert_ast(to_add2t(overflow.operand).side_2);
    operand[0] = z3_smt_downcast(o1)->e;
    operand[1] = z3_smt_downcast(o2)->e;
    width_op0 = to_add2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_add2t(overflow.operand).side_2->type->get_width();
    call1 = workaround_Z3_mk_bvadd_no_overflow;
    call2 = workaround_Z3_mk_bvadd_no_underflow;
    if (is_signedbv_type(to_add2t(overflow.operand).side_1) ||
        is_signedbv_type(to_add2t(overflow.operand).side_2))
      is_signed = Z3_L_TRUE;
  } else if (is_sub2t(overflow.operand)) {
    const smt_ast *o1 = convert_ast(to_sub2t(overflow.operand).side_1);
    const smt_ast *o2 = convert_ast(to_sub2t(overflow.operand).side_2);
    operand[0] = z3_smt_downcast(o1)->e;
    operand[1] = z3_smt_downcast(o2)->e;
    width_op0 = to_sub2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_sub2t(overflow.operand).side_2->type->get_width();
    call1 = workaround_Z3_mk_bvsub_no_underflow;
    call2 = workaround_Z3_mk_bvsub_no_overflow;
    if (is_signedbv_type(to_sub2t(overflow.operand).side_1) ||
        is_signedbv_type(to_sub2t(overflow.operand).side_2))
      is_signed = Z3_L_TRUE;
  } else if (is_mul2t(overflow.operand)) {
    const smt_ast *o1 = convert_ast(to_mul2t(overflow.operand).side_1);
    const smt_ast *o2 = convert_ast(to_mul2t(overflow.operand).side_2);
    operand[0] = z3_smt_downcast(o1)->e;
    operand[1] = z3_smt_downcast(o2)->e;
    width_op0 = to_mul2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_mul2t(overflow.operand).side_2->type->get_width();
    // XXX jmorse - no reference counting workaround for this; disassembling
    // these Z3 routines show that they've been touched by reference count
    // switchover, and so are likely actually reference counting correctly.
    call1 = Z3_mk_bvmul_no_overflow;
    call2 = Z3_mk_bvmul_no_underflow;
    if (is_signedbv_type(to_mul2t(overflow.operand).side_1) ||
        is_signedbv_type(to_mul2t(overflow.operand).side_2))
      is_signed = Z3_L_TRUE;
  } else {
    std::cerr << "Overflow operation with invalid operand";
    abort();
  }

  // XXX jmorse - int2bv trainwreck.
  if (int_encoding) {
    operand[0] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op0, operand[0]));
    operand[1] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op1, operand[1]));
  }

  result[0] = z3::to_expr(ctx, call1(z3_ctx, operand[0], operand[1], is_signed));
  result[1] = z3::to_expr(ctx, call2(z3_ctx, operand[0], operand[1]));
  output = !(result[0] && result[1]);
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast(output, s, expr);
}

smt_ast *
z3_convt::overflow_cast(const expr2tc &expr)
{
  const overflow_cast2t &ocast = to_overflow_cast2t(expr);
  z3::expr output;
  uint64_t result;
  u_int width;

  width = ocast.operand->type->get_width();

  if (ocast.bits >= width || ocast.bits == 0)
    throw new conv_error("overflow-typecast got wrong number of bits");

  assert(ocast.bits <= 32 && ocast.bits != 0);
  result = 1 << ocast.bits;

  expr2tc oper = ocast.operand;

  // Cast fixedbv to its integer form.
  if (is_fixedbv_type(ocast.operand)) {
    const fixedbv_type2t &fbvt = to_fixedbv_type(ocast.operand->type);
    type2tc signedbv(new signedbv_type2t(fbvt.integer_bits));
    oper = typecast2tc(signedbv, oper);
  }

  expr2tc lessthan, greaterthan;
  if (is_signedbv_type(ocast.operand) ||
      is_fixedbv_type(ocast.operand)) {
    // Produce some useful constants
    unsigned int nums_width = (is_signedbv_type(ocast.operand))
                               ? width : width / 2;
    type2tc signedbv(new signedbv_type2t(nums_width));

    constant_int2tc result_val = gen_uint(result / 2);
    constant_int2tc two = gen_uint(2);
    constant_int2tc minus_one(signedbv, BigInt(-1));

    // Now produce numbers that bracket the selected bitwidth. So for 16 bis
    // we would generate 2^15-1 and -2^15
    sub2tc upper(signedbv, result_val, minus_one);
    mul2tc lower(signedbv, result_val, minus_one);

    // Ensure operand lies between these braces
    lessthan = lessthan2tc(oper, upper);
    greaterthan = greaterthan2tc(oper, lower);
  } else if (is_unsignedbv_type(ocast.operand)) {
    // Create zero and 2^bitwidth,
    type2tc unsignedbv(new unsignedbv_type2t(width));

    constant_int2tc zero = zero_uint;
    constant_int2tc the_width = gen_uint(result);

    // Ensure operand lies between those numbers.
    lessthan = lessthan2tc(oper, the_width);
    greaterthan = greaterthanequal2tc(oper, zero);
  }

  z3::expr ops[2];
  const z3_smt_ast *tmp1, *tmp2;
  tmp1 = z3_smt_downcast(convert_ast(lessthan));
  ops[0] = tmp1->e;
  tmp2 = z3_smt_downcast(convert_ast(greaterthan));
  ops[1] = tmp2->e;

  output = !(ops[0] && ops[1]);
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast(output, s, expr);
}

smt_ast *
z3_convt::overflow_neg(const expr2tc &expr)
{
  const overflow_neg2t &neg = to_overflow_neg2t(expr);
  z3::expr output, operand;
  unsigned width;

  const z3_smt_ast *tmpast = z3_smt_downcast(convert_ast(neg.operand));
  operand = tmpast->e;

  // XXX jmorse - clearly wrong. Neg of pointer?
  if (is_pointer_type(neg.operand))
    operand = mk_tuple_select(operand, 1);

  width = neg.operand->type->get_width();

  // XXX jmorse - int2bv trainwreck
  if (int_encoding)
    operand = to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, operand));

  z3::expr no_over = z3::to_expr(ctx,
                           workaround_Z3_mk_bvneg_no_overflow(z3_ctx, operand));
  output = z3::to_expr(ctx, Z3_mk_not(z3_ctx, no_over));
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new z3_smt_ast(output, s, expr);
}

// Gigantic hack, implement a method in z3::ast, so that we can call from gdb
namespace z3 {
  void ast::dump(void) const {
    std::cout << Z3_ast_to_string(ctx(), m_ast) << std::endl;
    std::cout << "sort is " << Z3_sort_to_string(ctx(), Z3_get_sort(ctx(), m_ast)) << std::endl;
  }
};

// Caution: these workaround functions are a derived work from disassembling
// Z3 and adding reference counting to them. Technically they're derivative
// works that the microsoft research license demands be licensed under the
// microsoft research license, which puts us in a tricky copyright situation.
//
// Section 50C of the copyright act states that I can fix their software if
// necessary, but says nothing of distribution, and says that I can't ignore
// terms in MS' license restricting its lawful use.
//
// So, don't distribute this.
#ifdef NDEBUG
#error Don't distribute/release shadily licensed MS workaround code. And don't delete this error without asking jmorse.
#endif

Z3_ast
workaround_Z3_mk_bvadd_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                   Z3_bool is_signed)
{

  if (is_signed == Z3_L_TRUE) {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    Z3_ast zero = Z3_mk_int(ctx, 0, s);
    Z3_inc_ref(ctx, zero);
    Z3_ast add = Z3_mk_bvadd(ctx, a1, a2);
    Z3_inc_ref(ctx, add);
    Z3_ast lt1 = Z3_mk_bvslt(ctx, zero, a1);
    Z3_inc_ref(ctx, lt1);
    Z3_ast lt2 = Z3_mk_bvslt(ctx, zero, a2);
    Z3_inc_ref(ctx, lt2);
    Z3_ast args[2] = { lt1, lt2 };
    Z3_ast theand = Z3_mk_and(ctx, 2, args);
    Z3_inc_ref(ctx, theand);
    Z3_ast lt3 = Z3_mk_bvslt(ctx, zero, add);
    Z3_inc_ref(ctx, lt3);
    Z3_ast imp = Z3_mk_implies(ctx, theand, lt3);
    Z3_dec_ref(ctx, lt3);
    Z3_dec_ref(ctx, theand);
    Z3_dec_ref(ctx, lt2);
    Z3_dec_ref(ctx, lt1);
    Z3_dec_ref(ctx, add);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return imp;
  } else {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    unsigned int sort_size = Z3_get_bv_sort_size(ctx, s);
    Z3_ast ext1 = Z3_mk_zero_ext(ctx, 1, a1);
    Z3_inc_ref(ctx, ext1);
    Z3_ast ext2 = Z3_mk_zero_ext(ctx, 1, a2);
    Z3_inc_ref(ctx, ext2);
    Z3_ast add = Z3_mk_bvadd(ctx, ext1, ext2);
    Z3_inc_ref(ctx, add);
    Z3_sort s2 = Z3_mk_bv_sort(ctx, 1);
    Z3_inc_ref(ctx, (Z3_ast)s2);
    Z3_ast zero = Z3_mk_int(ctx, 0, s2);
    Z3_inc_ref(ctx, zero);
    Z3_ast ext = Z3_mk_extract(ctx, sort_size, sort_size, add);
    Z3_inc_ref(ctx, ext);
    Z3_ast eq = Z3_mk_eq(ctx, ext, zero);
    Z3_dec_ref(ctx, ext);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s2);
    Z3_dec_ref(ctx, add);
    Z3_dec_ref(ctx, ext2);
    Z3_dec_ref(ctx, ext1);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return eq;
  }
}

Z3_ast
workaround_Z3_mk_bvadd_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2)
{
  Z3_sort s = Z3_get_sort(ctx, a1);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast zero = Z3_mk_int(ctx, 0, s);
  Z3_inc_ref(ctx, zero);
  Z3_ast add = Z3_mk_bvadd(ctx, a1, a2);
  Z3_inc_ref(ctx, add);
  Z3_ast lt1 = Z3_mk_bvslt(ctx, a1, zero);
  Z3_inc_ref(ctx, lt1);
  Z3_ast lt2 = Z3_mk_bvslt(ctx, a2, zero);
  Z3_inc_ref(ctx, lt2);
  Z3_ast args[2] = { lt1, lt2 };
  Z3_ast theand = Z3_mk_and(ctx, 2, args);
  Z3_inc_ref(ctx, theand);
  Z3_ast lt3 = Z3_mk_bvslt(ctx, add, zero);
  Z3_inc_ref(ctx, lt3);
  Z3_ast imp = Z3_mk_implies(ctx, theand, lt3);
  Z3_dec_ref(ctx, lt3);
  Z3_dec_ref(ctx, theand);
  Z3_dec_ref(ctx, lt2);
  Z3_dec_ref(ctx, lt1);
  Z3_dec_ref(ctx, add);
  Z3_dec_ref(ctx, zero);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return imp;
}

Z3_ast
workaround_Z3_mk_bvsub_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                    Z3_bool is_signed)
{

  if (is_signed == Z3_L_TRUE) {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    Z3_ast zero = Z3_mk_int(ctx, 0, s);
    Z3_inc_ref(ctx, zero);
    Z3_ast neg = Z3_mk_bvneg(ctx, a2);
    Z3_inc_ref(ctx, neg);
    Z3_ast no_under = workaround_Z3_mk_bvadd_no_underflow(ctx, a1, neg);
    Z3_inc_ref(ctx, no_under);
    Z3_ast lt1 = Z3_mk_bvslt(ctx, zero, a2);
    Z3_inc_ref(ctx, lt1);
    Z3_ast imp = Z3_mk_implies(ctx, lt1, no_under);
    Z3_dec_ref(ctx, lt1);
    Z3_dec_ref(ctx, no_under);
    Z3_dec_ref(ctx, neg);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return imp;
  } else {
    return Z3_mk_bvule(ctx, a2, a1);
  }
}

extern "C" Z3_ast Z3_mk_bvsmin(Z3_context, Z3_sort);

Z3_ast
workaround_Z3_mk_bvsub_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2)
{

  Z3_sort s = Z3_get_sort(ctx, a2);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast neg = Z3_mk_bvneg(ctx, a2);
  Z3_inc_ref(ctx, neg);
//  Z3_ast min = Z3_mk_bvsmin(ctx, s);
//  Z3_inc_ref(ctx, min);
  Z3_ast min;
  {
    unsigned int width = Z3_get_bv_sort_size(ctx, s);
    Z3_ast sz = Z3_mk_int64(ctx, width - 1, s);
    Z3_inc_ref(ctx, sz);
    Z3_ast one = Z3_mk_int64(ctx, 1, s);
    Z3_inc_ref(ctx, one);
    Z3_ast msb = Z3_mk_bvshl(ctx, one, sz);
    Z3_inc_ref(ctx, msb);
    min = msb;
    Z3_dec_ref(ctx, one);
    Z3_dec_ref(ctx, sz);
  }
  Z3_ast no_over = workaround_Z3_mk_bvadd_no_overflow(ctx, a1, neg, 1);
  Z3_inc_ref(ctx, no_over);
  Z3_ast zero = Z3_mk_int(ctx, 0, s);
  Z3_inc_ref(ctx, zero);
  Z3_ast lt = Z3_mk_bvslt(ctx, a1, zero);
  Z3_inc_ref(ctx, lt);
  Z3_ast eq = Z3_mk_eq(ctx, a2, min);
  Z3_inc_ref(ctx, eq);
  Z3_ast ite = Z3_mk_ite(ctx, eq, lt, no_over);
  Z3_dec_ref(ctx, eq);
  Z3_dec_ref(ctx, lt);
  Z3_dec_ref(ctx, zero);
  Z3_dec_ref(ctx, no_over);
  Z3_dec_ref(ctx, min);
  Z3_dec_ref(ctx, neg);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return ite;
}

Z3_ast
workaround_Z3_mk_bvneg_no_overflow(Z3_context ctx, Z3_ast a)
{

  Z3_sort s = Z3_get_sort(ctx, a);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast min;
  {
    unsigned int width = Z3_get_bv_sort_size(ctx, s);
    Z3_ast sz = Z3_mk_int64(ctx, width - 1, s);
    Z3_inc_ref(ctx, sz);
    Z3_ast one = Z3_mk_int64(ctx, 1, s);
    Z3_inc_ref(ctx, one);
    Z3_ast msb = Z3_mk_bvshl(ctx, one, sz);
    Z3_inc_ref(ctx, msb);
    min = msb;
    Z3_dec_ref(ctx, one);
    Z3_dec_ref(ctx, sz);
  }
  Z3_ast eq = Z3_mk_eq(ctx, a, min);
  Z3_inc_ref(ctx, eq);
  Z3_ast thenot = Z3_mk_not(ctx, eq);
  Z3_dec_ref(ctx, eq);
  Z3_dec_ref(ctx, min);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return thenot;
}
