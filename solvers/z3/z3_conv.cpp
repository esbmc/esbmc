/*******************************************************************
   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <assert.h>
#include <ctype.h>
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
#include <type_byte_size.h>
#include <prefix.h>
#include <fixedbv.h>
#include <base_type.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                          "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

smt_convt *
create_new_z3_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                              const optionst &opts __attribute__((unused)),
                              tuple_iface **tuple_api, array_iface **array_api)
{
  z3_convt *conv = new z3_convt(int_encoding, is_cpp, ns);
  *tuple_api = static_cast<tuple_iface*>(conv);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

Z3_ast workaround_Z3_mk_bvadd_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvadd_no_underflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_overflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvneg_no_overflow(Z3_context ctx, Z3_ast a);
z3_convt::z3_convt(bool int_encoding, bool is_cpp, const namespacet &_ns)
: smt_convt(int_encoding, _ns, is_cpp), array_iface(true, true),ctx(false)
{

  this->int_encoding = int_encoding;

  assumpt_mode = false;

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

  assumpt_ctx_stack.push_back(assumpt.begin());

  z3_convt::init_addr_space_array();
}

z3_convt::~z3_convt()
{
  delete_all_asts();
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
  pointer_logic.push_back(pointer_logic.back());
  addr_space_sym_num.push_back(addr_space_sym_num.back());
  addr_space_data.push_back(addr_space_data.back());

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
}

void
z3_convt::init_addr_space_array(void)
{

  convert_type(addr_space_type, addr_space_tuple_sort);
  Z3_func_decl tmp_addr_space_decl =
    Z3_get_tuple_sort_mk_decl(ctx, addr_space_tuple_sort);
  addr_space_tuple_decl = z3::func_decl(ctx, tmp_addr_space_decl);

  addr_space_arr_sort = 
                  ctx.array_sort(ctx.esbmc_int_sort(), addr_space_tuple_sort);

  return;
}

smt_convt::resultt
z3_convt::dec_solve(void)
{
  unsigned major, minor, build, revision;
  z3::check_result result;
  Z3_get_version(&major, &minor, &build, &revision);

  pre_solve();

  result = check2_z3_properties();

  if (result == z3::unsat)
    return smt_convt::P_UNSATISFIABLE;
  else if (result == z3::unknown)
    return smt_convt::P_ERROR;
  else
    return smt_convt::P_SATISFIABLE;
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
                                    z3::sort &sort)
{
  z3::symbol mk_tuple_name, *proj_names;
  z3::sort *proj_types;
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
                              false, sort);
    break;
  }
  case type2t::union_id:
  {
    const union_type2t &uni = to_union_type(type);
    convert_struct_union_type(uni.members, uni.member_names, uni.name,
                              true, sort);
    break;
  }
  case type2t::array_id:
  {
    // Because of crazy domain sort rewriting, pass this via all the other smt
    // processing code.
    const array_type2t &arr = to_array_type(type);
    unsigned int domain_width = calculate_array_domain_width(arr);

    smt_sortt domain;
    if (int_encoding)
      domain = mk_sort(SMT_SORT_INT);
    else
      domain = mk_sort(SMT_SORT_BV, domain_width, false);

    smt_sortt range = convert_sort(arr.subtype);
    sort = z3_sort_downcast(mk_sort(SMT_SORT_ARRAY, domain, range))->s;
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

tvt
z3_convt::l_get(const smt_ast *a)
{
  tvt result = tvt(tvt::TV_ASSUME);

  expr2tc res = get_bool(a);

  if (!is_nil_expr(res) && is_constant_bool2t(res)) {
    result = (to_constant_bool2t(res).is_true())
             ? tvt(tvt::TV_TRUE) : tvt(tvt::TV_FALSE);
  } else {
    result = tvt(tvt::TV_UNKNOWN);
  }


  return result;
}

void
z3_convt::assert_ast(const smt_ast *a)
{
  const z3_smt_ast *za = z3_smt_downcast(a);
  z3::expr theval = za->e;
  solver.add(theval);
  assumpt.push_back(theval);
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

  z3::expr newvar = ctx.fresh_const("", ctx.bool_sort());
  z3::expr formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, newvar, ast));
  solver.add(formula);

  assumpt.push_back(newvar);

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

  std::vector<z3::expr> new_fields;
  new_fields.resize(num_fields);
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

  return mk_tuple_decl.make_tuple_from_array(num_fields, new_fields.data());
}

z3::expr
z3_convt::mk_tuple_select(const z3::expr &t, unsigned i)
{
  z3::sort ty;
  unsigned num_fields;

  ty = t.get_sort();

  if (!ty.is_datatype()) {
    std::cerr << "Z3 conversion: argument must be a tuple" << std::endl;
    abort();
  }

  num_fields = Z3_get_tuple_sort_num_fields(ctx, ty);

  if (i >= num_fields) {
    std::cerr << "Z3 conversion: invalid tuple select, index is too large"
              << std::endl;
    abort();
  }

  z3::func_decl proj_decl =
    z3::to_func_decl(ctx, Z3_get_tuple_sort_field_decl(ctx, ty, i));
  return proj_decl(t);
}

// SMT-abstraction migration routines.

smt_astt
z3_convt::mk_func_app(const smt_sort *s, smt_func_kind k,
                      const smt_ast * const *args,
                      unsigned int numargs)
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
    return new_ast(mk_add(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
    return new_ast(mk_sub(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
    return new_ast((asts[0]->e * asts[1]->e), s);
  case SMT_FUNC_MOD:
    return new_ast(
                    z3::to_expr(ctx, Z3_mk_mod(z3_ctx, asts[0]->e, asts[1]->e)),
                    s);
  case SMT_FUNC_BVSMOD:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvsrem(z3_ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_BVUMOD:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvurem(z3_ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_DIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_div(z3_ctx, asts[0]->e, asts[1]->e)),s);
  case SMT_FUNC_BVSDIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvsdiv(z3_ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_BVUDIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvudiv(z3_ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_SHL:
    return new_ast(asts[0]->e * pw(ctx.int_val(2), asts[1]->e), s);
  case SMT_FUNC_BVSHL:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvshl(ctx, asts[0]->e, asts[1]->e)), s);
  case SMT_FUNC_BVASHR:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvashr(ctx, asts[0]->e, asts[1]->e)),s);
  case SMT_FUNC_NEG:
  case SMT_FUNC_BVNEG:
    return new_ast((-asts[0]->e), s);
  case SMT_FUNC_BVLSHR:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvlshr(ctx, asts[0]->e, asts[1]->e)),s);
  case SMT_FUNC_BVNOT:
    return new_ast((~asts[0]->e), s);
  case SMT_FUNC_BVNXOR:
    return new_ast(mk_bvxnor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_BVNOR:
    return new_ast(mk_bvnor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_BVNAND:
    return new_ast(mk_bvnor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_BVXOR:
    return new_ast(mk_bvxor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_BVOR:
    return new_ast(mk_bvor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_BVAND:
    return new_ast(mk_bvand(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_IMPLIES:
    return new_ast(mk_implies(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_XOR:
    return new_ast(mk_xor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_OR:
    return new_ast(mk_or(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_AND:
    return new_ast(mk_and(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_NOT:
    return new_ast(!asts[0]->e, s);
  // NB: mk_{l,g}t{,e} ignore unsigned arg in integer mode.
  case SMT_FUNC_LT:
  case SMT_FUNC_BVULT:
    return new_ast(mk_lt(asts[0]->e, asts[1]->e, true), s);
  case SMT_FUNC_BVSLT:
    return new_ast(mk_lt(asts[0]->e, asts[1]->e, false), s);
  case SMT_FUNC_GT:
  case SMT_FUNC_BVUGT:
    return new_ast(mk_gt(asts[0]->e, asts[1]->e, true), s);
  case SMT_FUNC_BVSGT:
    return new_ast(mk_gt(asts[0]->e, asts[1]->e, false), s);
  case SMT_FUNC_LTE:
  case SMT_FUNC_BVULTE:
    return new_ast(mk_le(asts[0]->e, asts[1]->e, true), s);
  case SMT_FUNC_BVSLTE:
    return new_ast(mk_le(asts[0]->e, asts[1]->e, false), s);
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVUGTE:
    return new_ast(mk_ge(asts[0]->e, asts[1]->e, true), s);
  case SMT_FUNC_BVSGTE:
    return new_ast(mk_ge(asts[0]->e, asts[1]->e, false), s);
  case SMT_FUNC_EQ:
    return new_ast((asts[0]->e == asts[1]->e), s);
  case SMT_FUNC_NOTEQ:
    return new_ast((asts[0]->e != asts[1]->e), s);
  case SMT_FUNC_ITE:
    return new_ast(ite(asts[0]->e, asts[1]->e, asts[2]->e), s);
  case SMT_FUNC_STORE:
    return new_ast(store(asts[0]->e, asts[1]->e, asts[2]->e), s);
  case SMT_FUNC_SELECT:
    return new_ast(select(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_CONCAT:
    return new_ast(
                   z3::to_expr(ctx, Z3_mk_concat(ctx, asts[0]->e, asts[1]->e)),
                   s);
  case SMT_FUNC_REAL2INT:
    return new_ast(z3::to_expr(ctx, Z3_mk_real2int(ctx, asts[0]->e)), s);
  case SMT_FUNC_INT2REAL:
    return new_ast(z3::to_expr(ctx, Z3_mk_int2real(ctx, asts[0]->e)), s);
  case SMT_FUNC_IS_INT:
    return new_ast(z3::to_expr(ctx, Z3_mk_is_int(ctx, asts[0]->e)), s);
  default:
    std::cerr << "Unhandled SMT func in z3 conversion" << std::endl;
    abort();
  }
}

smt_astt
z3_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low,
                     const smt_sort *s)
{

  return new_ast(z3::to_expr(ctx, Z3_mk_extract(ctx, high, low,
                                         z3_smt_downcast(a)->e)), s);
}

smt_astt
z3_convt::mk_smt_int(const mp_integer &theint, bool sign)
{
  smt_sort *s = mk_sort(SMT_SORT_INT, sign);
  if (theint.is_negative())
    return new_ast(ctx.int_val(theint.to_int64()), s);
  else
    return new_ast(ctx.int_val(theint.to_uint64()), s);
}

smt_astt
z3_convt::mk_smt_real(const std::string &str)
{
  smt_sort *s = mk_sort(SMT_SORT_REAL);
  return new_ast(ctx.real_val(str.c_str()), s);
}

smt_astt
z3_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int width)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, width, sign);
  if (theint.is_negative())
    return new_ast(ctx.bv_val(theint.to_int64(), width), s);
  else
    return new_ast(ctx.bv_val(theint.to_uint64(), width), s);
}

smt_astt
z3_convt::mk_smt_bool(bool val)
{
  smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new_ast(ctx.bool_val(val), s);
}

smt_astt
z3_convt::mk_array_symbol(const std::string &name, const smt_sort *s,
                          smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_astt
z3_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);
  return new_ast(ctx.constant(name.c_str(), zs->s), s);
}

smt_sort *
z3_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;
  z3_smt_sort *s = NULL, *dom, *range;
  unsigned long uint;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    s = new z3_smt_sort(k, ctx.int_sort(), 0);
    break;
  case SMT_SORT_REAL:
    s = new z3_smt_sort(k, ctx.real_sort());
    break;
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    s = new z3_smt_sort(k, ctx.bv_sort(uint), uint);
    break;
  case SMT_SORT_ARRAY:
  {
    dom = va_arg(ap, z3_smt_sort *); // Consider constness?
    range = va_arg(ap, z3_smt_sort *);
    assert(int_encoding || dom->data_width != 0);

    // The range data width is allowed to be zero, which happens if the range
    // is not a bitvector / integer
    unsigned int data_width = range->data_width;
    if (range->id == SMT_SORT_STRUCT || range->id == SMT_SORT_BOOL || range->id == SMT_SORT_UNION)
      data_width = 1;

    s = new z3_smt_sort(k, ctx.array_sort(dom->s, range->s), data_width,
                        dom->data_width, range);
    break;
  }
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

  if (is_array_type(type)) {
    const array_type2t &arrtype = to_array_type(type);
    unsigned int domain_width;
    if (int_encoding)
      domain_width = 0;
    else
      domain_width = s.array_domain().bv_size();

    // The '1' range is a dummy, seeing how smt_sortt has no representation of
    // tuple sort ranges
    return new z3_smt_sort(SMT_SORT_ARRAY, s, 1, domain_width,
                           convert_sort(arrtype.subtype));
  } else {
    return new z3_smt_sort(SMT_SORT_STRUCT, s, type);
  }
}

smt_sort *
z3_convt::mk_union_sort(const type2tc &type)
{
  z3::sort s;
  convert_type(type, s);
  return new z3_smt_sort(SMT_SORT_UNION, s, type);
}

const smt_ast *
z3_convt::z3_smt_ast::eq(smt_convt *ctx, const smt_ast *other) const
{
  const smt_sort *boolsort = ctx->mk_sort(SMT_SORT_BOOL);
  const smt_ast *args[2];
  args[0] = this;
  args[1] = other;
  return ctx->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
}

const smt_ast *
z3_convt::z3_smt_ast::update(smt_convt *ctx, const smt_ast *value,
                   unsigned int idx, expr2tc idx_expr) const
{

  expr2tc index;

  if (sort->id == SMT_SORT_ARRAY) {
    if (is_nil_expr(idx_expr)) {
      index = constant_int2tc(type2tc(new unsignedbv_type2t(sort->domain_width)),
            BigInt(idx));
    } else {
      index = idx_expr;
    }

    const smt_ast *args[3];
    args[0] = this;
    args[1] = ctx->convert_ast(index);
    args[2] = value;
    return ctx->mk_func_app(args[0]->sort, SMT_FUNC_STORE, args, 3);
  } else {
    assert(sort->id == SMT_SORT_STRUCT || sort->id == SMT_SORT_UNION);
    assert(is_nil_expr(idx_expr) &&
           "Can only update constant index tuple elems");

    z3_convt *z3_ctx = static_cast<z3_convt*>(ctx);
    const z3_smt_ast *updateval = z3_smt_downcast(value);
    return z3_ctx->new_ast(z3_ctx->mk_tuple_update(e, idx, updateval->e), sort);
  }
}

const smt_ast *
z3_convt::z3_smt_ast::select(smt_convt *ctx, const expr2tc &idx) const
{
  const smt_ast *args[2];
  args[0] = this;
  args[1] = ctx->convert_ast(idx);
  const smt_sort *rangesort = z3_sort_downcast(sort)->rangesort;
  return ctx->mk_func_app(rangesort, SMT_FUNC_SELECT, args, 2);
}

const smt_ast *
z3_convt::z3_smt_ast::project(smt_convt *ctx, unsigned int elem) const
{
  z3_convt *z3_ctx = static_cast<z3_convt*>(ctx);

  const z3_smt_sort *thesort = z3_sort_downcast(sort);
  assert(!is_nil_type(thesort->tupletype));
  const struct_union_data &data = ctx->get_type_def(thesort->tupletype);
  assert(elem < data.members.size());
  const smt_sort *idx_sort = ctx->convert_sort(data.members[elem]);

  return z3_ctx->new_ast(z3_ctx->mk_tuple_select(e, elem), idx_sort);
}

smt_astt
z3_convt::tuple_create(const expr2tc &structdef)
{
  z3::expr e;
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type =
    static_cast<const struct_union_data &>(*strct.type);

  convert_struct_union(strct.datatype_members, type.members, strct.type, e);
  smt_sort *s = mk_struct_sort(structdef->type);
  return new_ast(e, s);
}

smt_astt
z3_convt::union_create(const expr2tc &unidef __attribute__((unused)))
{
  std::cerr << "Union create in z3_convt called" << std::endl;
  abort();
}

smt_astt
z3_convt::tuple_fresh(const smt_sort *s, std::string name)
{
  const z3_smt_sort *zs = static_cast<const z3_smt_sort*>(s);
  const char *n = (name == "") ? NULL : name.c_str();
  z3::expr output = ctx.fresh_const(n, zs->s);
  return new_ast(output, zs);
}

const smt_ast *
z3_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  z3::sort dom_sort =
    (int_encoding)? ctx.int_sort() : ctx.bv_sort(domain_width);
  const z3_smt_sort *range = z3_sort_downcast(init_val->sort);
  z3::sort range_sort = range->s;
  z3::sort array_sort = ctx.array_sort(dom_sort, range_sort);

  z3::expr val = z3_smt_downcast(init_val)->e;
  z3::expr output = z3::to_expr(ctx, Z3_mk_const_array(ctx, dom_sort, val));

  long unsigned int range_width = range->data_width;
  if (range->id == SMT_SORT_STRUCT || range->id == SMT_SORT_BOOL || range->id == SMT_SORT_UNION)
    range_width = 1;

  long unsigned int dom_width = (int_encoding) ? 0 : dom_sort.bv_size();
  smt_sort *s =
    new z3_smt_sort(SMT_SORT_ARRAY, array_sort, range_width, dom_width, range);
  return new_ast(output, s);
}

const smt_ast *
z3_convt::tuple_array_create(const type2tc &arr_type,
                              const smt_ast **input_args, bool const_array,
                              const smt_sort *domain)
{
  z3::expr output;
  const array_type2t &arrtype = to_array_type(arr_type);

  if (const_array) {
    z3::expr value, index;
    z3::sort array_type, dom_type;
    std::string tmp, identifier;

    array_type = z3_sort_downcast(convert_sort(arr_type))->s;
    dom_type = array_type.array_domain();

    const z3_smt_ast *tmpast = z3_smt_downcast(*input_args);
    value = tmpast->e;

    if (is_bool_type(arrtype.subtype)) {
      value = ctx.bool_val(false);
    }

    output = z3::to_expr(ctx, Z3_mk_const_array(ctx, dom_type, value));
  } else {
    u_int i = 0;
    z3::sort z3_array_type;
    z3::expr int_cte, val_cte;
    z3::sort domain_sort;

    assert(!is_nil_expr(arrtype.array_size) && "Non-const array-of's can't be infinitely sized");
    const constant_int2t &sz = to_constant_int2t(arrtype.array_size);

    assert(is_constant_int2t(arrtype.array_size) &&
           "array_of sizes should be constant");

    int64_t size;
    size = sz.as_long();

    z3_array_type = z3_sort_downcast(convert_sort(arr_type))->s;
    domain_sort = z3_array_type.array_domain();

    output = ctx.fresh_const(NULL, z3_array_type);

    for (i = 0; i < size; i++) {
      int_cte = ctx.num_val(i, domain_sort);
      const z3_smt_ast *tmpast = z3_smt_downcast(input_args[i]);
      output = z3::store(output, int_cte, tmpast->e);
    }
  }

  smt_sort *ssort = mk_struct_sort(arrtype.subtype);
  smt_sort *asort = mk_sort(SMT_SORT_ARRAY, domain, ssort);
  return new_ast(output, asort);
}

smt_astt
z3_convt::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_astt
z3_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
}

smt_astt
z3_convt::tuple_array_of(const expr2tc &init, unsigned long domain_width)
{
  return convert_array_of(convert_ast(init), domain_width);
}

expr2tc
z3_convt::tuple_get(const expr2tc &expr)
{
  const struct_union_data &strct = get_type_def(expr->type);

  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  forall_types(it, strct.members) {
    member2tc memb(*it, expr, strct.member_names[i]);
    outstruct.get()->datatype_members.push_back(get(memb));
    i++;
  }

  // If it's a pointer, rewrite.
  if (is_pointer_type(expr->type)) {
    uint64_t num = to_constant_int2t(outstruct->datatype_members[0])
                                    .constant_value.to_uint64();
    uint64_t offs = to_constant_int2t(outstruct->datatype_members[1])
                                     .constant_value.to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  return outstruct;
}

const smt_ast *
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
  return new_ast(output, s);
}

smt_ast *
z3_convt::overflow_cast(const expr2tc &expr)
{
  const overflow_cast2t &ocast = to_overflow_cast2t(expr);
  z3::expr output;
  uint64_t result;
  u_int width;

  width = ocast.operand->type->get_width();

  if (ocast.bits >= width || ocast.bits == 0) {
    std::cerr << "Z3 conversion: overflow-typecast got wrong number of bits"
              << std::endl;
    abort();
  }

  assert(ocast.bits <= 32 && ocast.bits != 0);
  result = 1ULL << ocast.bits;

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
    type2tc unsignedbv(new unsignedbv_type2t(nums_width));

    constant_int2tc result_val =
      constant_int2tc(unsignedbv, BigInt(result / 2));
    constant_int2tc two =
      constant_int2tc(unsignedbv, BigInt(2));
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

    constant_int2tc zero = gen_uint(unsignedbv, 0);
    constant_int2tc the_width = gen_uint(unsignedbv, result);

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
  return new_ast(output, s);
}

const smt_ast *
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
  return new_ast(output, s);
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

// ***************************** 'get' api *******************************

expr2tc
z3_convt::get_bool(const smt_ast *a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  const z3_smt_ast *za = z3_smt_downcast(a);

  z3::expr e = za->e;
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  if (Z3_get_bool_value(z3_ctx, e) == Z3_L_TRUE)
    return true_expr;
  else
    return false_expr;
}

expr2tc
z3_convt::get_bv(const type2tc &t, const smt_ast *a)
{
  const z3_smt_ast *za = z3_smt_downcast(a);

  z3::expr e = za->e;
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  if (Z3_get_ast_kind(z3_ctx, e) != Z3_NUMERAL_AST)
    return expr2tc();

  std::string value = Z3_get_numeral_string(z3_ctx, e);
  return constant_int2tc(t, BigInt(value.c_str()));
}

expr2tc
z3_convt::get_array_elem(const smt_ast *array, uint64_t index,
                         const type2tc &subtype)
{
  const z3_smt_ast *za = z3_smt_downcast(array);
  unsigned long bv_size = array->sort->domain_width;
  const z3_smt_ast *idx;
  if (int_encoding)
    idx = static_cast<const z3_smt_ast*>(mk_smt_int(BigInt(index), false));
  else
    idx = static_cast<const z3_smt_ast*>(mk_smt_bvint(BigInt(index), false, bv_size));

  z3::expr e = select(za->e, idx->e);
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  z3_smt_ast *value = new_ast(e, convert_sort(subtype));
  type2tc res_type = (int_encoding) ? get_int_type(64) : get_uint_type(bv_size);
  expr2tc result = get_bv(res_type, value);

  return result;
}

void
z3_convt::debug_label_formula(std::string name, const z3::expr &formula)
{
  std::stringstream ss;
  unsigned &num = debug_label_map[name];
  ss << "__ESBMC_" << name << num;
  std::string the_name = ss.str();
  num++;

  z3::expr sym = ctx.constant(the_name.c_str(), formula.get_sort());
  z3::expr eq = sym == formula;
  assert_formula(eq);
  return;
}

const smt_ast *
z3_convt::make_disjunct(const ast_vec &v)
{
  // Make a gigantic 'or'.
  Z3_ast arr[v.size()];

  size_t i = 0;
  for (ast_vec::const_iterator it = v.begin(); it != v.end(); it++, i++)
    arr[i] = z3_smt_downcast(*it)->e;

  z3::expr e = z3::to_expr(ctx, Z3_mk_or(z3_ctx, v.size(), arr));
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new_ast(e, s);
}

const smt_ast *
z3_convt::make_conjunct(const ast_vec &v)
{
  // Make a gigantic 'and'.
  Z3_ast arr[v.size()];

  size_t i = 0;
  for (ast_vec::const_iterator it = v.begin(); it != v.end(); it++, i++)
    arr[i] = z3_smt_downcast(*it)->e;

  z3::expr e = z3::to_expr(ctx, Z3_mk_and(z3_ctx, v.size(), arr));
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  return new_ast(e, s);
}

void
z3_convt::add_array_constraints_for_solving()
{
  return;
}

void
z3_convt::push_array_ctx(void)
{
  return;
}

void
z3_convt::pop_array_ctx(void)
{
  return;
}

void
z3_convt::add_tuple_constraints_for_solving()
{
  return;
}

void
z3_convt::push_tuple_ctx()
{
  return;
}

void
z3_convt::pop_tuple_ctx()
{
  return;
}
