/*******************************************************************
   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <cassert>
#include <cctype>
#include <fstream>
#include <sstream>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/string2array.h>
#include <util/type_byte_size.h>
#include <z3_conv.h>

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                          "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

smt_convt *
create_new_z3_solver(bool int_encoding, const namespacet &ns,
                              const optionst &opts __attribute__((unused)),
                              tuple_iface **tuple_api, array_iface **array_api)
{
  z3_convt *conv = new z3_convt(int_encoding, ns);
  *tuple_api = static_cast<tuple_iface*>(conv);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

z3_convt::z3_convt(bool int_encoding, const namespacet &_ns)
: smt_convt(int_encoding, _ns), array_iface(true, true),ctx(false)
{

  this->int_encoding = int_encoding;

  assumpt_mode = false;

  z3::config conf;
  ctx.init(conf, int_encoding);

  solver =
     (z3::tactic(ctx, "simplify") &
      z3::tactic(ctx, "solve-eqs") &
      z3::tactic(ctx, "simplify") &
      z3::tactic(ctx, "smt")).mk_solver();

  z3::params p(ctx);
  p.set("relevancy", (unsigned int) 0);
  solver.set(p);

  Z3_set_ast_print_mode(ctx, Z3_PRINT_SMTLIB_COMPLIANT);

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

  if (config.options.get_bool_option("show-smt-model") && result == z3::sat)
    std::cout << Z3_model_to_string(ctx, model);

  return result;
}

void
z3_convt::convert_struct_type(const std::vector<type2tc> &members,
                                    const std::vector<irep_idt> &member_names,
                                    const irep_idt &struct_name, z3::sort &sort)
{
  z3::symbol mk_tuple_name, *proj_names;
  z3::sort *proj_types;
  Z3_func_decl mk_tuple_decl, *proj_decls;
  std::string name;
  u_int num_elems;

  num_elems = members.size();

  proj_names = new z3::symbol[num_elems];
  proj_types = new z3::sort[num_elems];
  proj_decls = new Z3_func_decl[num_elems];

  name = "struct";
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
    const z3_smt_sort* tmp = z3_sort_downcast(convert_sort(*it));
    proj_types[i] = tmp->s;
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
z3_convt::convert_struct(const std::vector<expr2tc> &members,
                         const std::vector<type2tc> &member_types,
                         const type2tc &type, z3::expr &output)
{

  // Converts a static struct - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  u_int i = 0;

  z3::sort sort;
  convert_type(type, sort);

  unsigned size = member_types.size();

  z3::expr *args = new z3::expr[size];

#ifndef NDEBUG
  unsigned int numoperands = members.size();
  assert(numoperands == member_types.size() &&
         "Too many / few struct fields for struct type");
#endif

  // Populate tuple with members of that struct
  forall_types(it, member_types) {
    const z3_smt_ast *tmp = z3_smt_downcast(convert_ast(members[i]));
    args[i] = tmp->e;
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
    convert_struct_type(strct.members, strct.member_names, strct.name, sort);
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
  z3::expr formula = z3::to_expr(ctx, Z3_mk_iff(ctx, newvar, ast));
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

  switch (k) {
  case SMT_FUNC_ADD:
  case SMT_FUNC_BVADD:
    return new_ast((asts[0]->e + asts[1]->e), s);
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
    return new_ast((asts[0]->e - asts[1]->e), s);
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
    return new_ast((asts[0]->e * asts[1]->e), s);
  case SMT_FUNC_MOD:
    if(s->id == SMT_SORT_FLOATBV)
      return new_ast(z3::to_expr(ctx, Z3_mk_fpa_rem(ctx, asts[0]->e, asts[1]->e)), s);
    else
      return new_ast(z3::to_expr(ctx, Z3_mk_mod(ctx, asts[0]->e, asts[1]->e)), s);
  case SMT_FUNC_BVSMOD:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvsrem(ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_BVUMOD:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvurem(ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_DIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_div(ctx, asts[0]->e, asts[1]->e)),s);
  case SMT_FUNC_BVSDIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvsdiv(ctx, asts[0]->e, asts[1]->e)),
                 s);
  case SMT_FUNC_BVUDIV:
    return new_ast(
                 z3::to_expr(ctx, Z3_mk_bvudiv(ctx, asts[0]->e, asts[1]->e)),
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
    return new_ast(!(asts[0]->e ^ asts[1]->e), s);
  case SMT_FUNC_BVNOR:
    return new_ast(!(asts[0]->e | asts[1]->e), s);
  case SMT_FUNC_BVNAND:
    return new_ast(!(asts[0]->e & asts[1]->e), s);
  case SMT_FUNC_BVXOR:
    return new_ast((asts[0]->e ^ asts[1]->e), s);
  case SMT_FUNC_BVOR:
    return new_ast((asts[0]->e | asts[1]->e), s);
  case SMT_FUNC_BVAND:
    return new_ast((asts[0]->e & asts[1]->e), s);
  case SMT_FUNC_IMPLIES:
    return new_ast(implies(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_XOR:
    return new_ast(mk_xor(asts[0]->e, asts[1]->e), s);
  case SMT_FUNC_OR:
    return new_ast((asts[0]->e || asts[1]->e), s);
  case SMT_FUNC_AND:
    return new_ast((asts[0]->e && asts[1]->e), s);
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
  case SMT_FUNC_FABS:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_abs(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISNAN:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_nan(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISINF:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_infinite(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISNORMAL:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_normal(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISZERO:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_zero(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISNEG:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_negative(ctx, asts[0]->e)),s);
  case SMT_FUNC_ISPOS:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_is_positive(ctx, asts[0]->e)),s);
  case SMT_FUNC_IEEE_EQ:
    return new_ast(z3::to_expr(ctx, Z3_mk_fpa_eq(ctx, asts[0]->e, asts[1]->e)),s);
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
  case SMT_FUNC_BV2FLOAT:
    return new_ast(ctx.fpa_from_bv(asts[0]->e, z3_sort_downcast(s)->s), s);
  case SMT_FUNC_FLOAT2BV:
    return new_ast(ctx.fpa_to_ieeebv(asts[0]->e), s);
  default:
    std::cerr << "Unhandled SMT func in z3 conversion" << std::endl;
    abort();
  }
}

smt_astt
z3_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low,
                     const smt_sort *s)
{
  const z3_smt_ast *za = z3_smt_downcast(a);

  // If it's a floatbv, convert it to bv
  if(a->sort->id == SMT_SORT_FLOATBV)
  {
    smt_ast * bv = new_ast(ctx.fpa_to_ieeebv(za->e), s);
    za = z3_smt_downcast(bv);
  }

  return new_ast(z3::to_expr(ctx, Z3_mk_extract(ctx, high, low, za->e)), s);
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
z3_convt::mk_smt_bvfloat(const ieee_floatt &thereal,
                         unsigned ew, unsigned sw)
{
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);

  const mp_integer sig = thereal.get_fraction();

  // If the number is denormal, we set the exponent to -bias
  const mp_integer exp = thereal.is_normal() ?
    thereal.get_exponent() + thereal.spec.bias() : 0;

  smt_astt sgn_bv = mk_smt_bvint(BigInt(thereal.get_sign()), false, 1);
  smt_astt exp_bv = mk_smt_bvint(exp, false, ew);
  smt_astt sig_bv = mk_smt_bvint(sig, false, sw);

  return new_ast(
    ctx.fpa_val(
      z3_smt_downcast(sgn_bv)->e,
      z3_smt_downcast(exp_bv)->e,
      z3_smt_downcast(sig_bv)->e), s);
}

smt_astt z3_convt::mk_smt_bvfloat_nan(unsigned ew, unsigned sw)
{
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);

  return new_ast(ctx.fpa_nan(zs->s), s);
}

smt_astt z3_convt::mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);

  return new_ast(ctx.fpa_inf(sgn, zs->s), s);
}

smt_astt z3_convt::mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm)
{
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV_RM);

  switch(rm)
  {
    case ieee_floatt::ROUND_TO_EVEN:
      return new_ast(ctx.fpa_rm_ne(), s);
    case ieee_floatt::ROUND_TO_MINUS_INF:
      return new_ast(ctx.fpa_rm_mi(), s);
    case ieee_floatt::ROUND_TO_PLUS_INF:
      return new_ast(ctx.fpa_rm_pi(), s);
    case ieee_floatt::ROUND_TO_ZERO:
      return new_ast(ctx.fpa_rm_ze(), s);
    default:
      break;
  }

  abort();
}

smt_astt z3_convt::mk_smt_typecast_from_bvfloat(const typecast2t &cast)
{
  // Rounding mode symbol
  smt_astt rm_const;

  smt_astt from = convert_ast(cast.from);
  const z3_smt_ast *mfrom = z3_smt_downcast(from);

  smt_sort *s;
  if(is_unsignedbv_type(cast.type)) {
    s = mk_sort(SMT_SORT_BV);

    // Conversion from float to integers always truncate, so we assume
    // the round mode to be toward zero
    rm_const = mk_smt_bvfloat_rm(ieee_floatt::ROUND_TO_ZERO);
    const z3_smt_ast *mrm_const = z3_smt_downcast(rm_const);

    return new_ast(ctx.fpa_to_ubv(mrm_const->e, mfrom->e, cast.type->get_width()), s);
  } else if(is_signedbv_type(cast.type)) {
    s = mk_sort(SMT_SORT_BV);

    // Conversion from float to integers always truncate, so we assume
    // the round mode to be toward zero
    rm_const = mk_smt_bvfloat_rm(ieee_floatt::ROUND_TO_ZERO);
    const z3_smt_ast *mrm_const = z3_smt_downcast(rm_const);

    return new_ast(ctx.fpa_to_sbv(mrm_const->e, mfrom->e, cast.type->get_width()), s);
  } else if(is_floatbv_type(cast.type)) {
    unsigned ew = to_floatbv_type(cast.type).exponent;
    unsigned sw = to_floatbv_type(cast.type).fraction;

    s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
    const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);

    // Use the round mode
    rm_const = convert_rounding_mode(cast.rounding_mode);
    const z3_smt_ast *mrm_const = z3_smt_downcast(rm_const);

    return new_ast(ctx.fpa_to_fpa(mrm_const->e, mfrom->e, zs->s), s);
  }

  abort();
}

smt_astt z3_convt::mk_smt_typecast_to_bvfloat(const typecast2t &cast)
{
  // Rounding mode symbol
  smt_astt rm_const = convert_rounding_mode(cast.rounding_mode);
  const z3_smt_ast *mrm_const = z3_smt_downcast(rm_const);

  // Convert the expr to be casted
  smt_astt from = convert_ast(cast.from);
  const z3_smt_ast *mfrom = z3_smt_downcast(from);

  // The target type
  unsigned ew = to_floatbv_type(cast.type).exponent;
  unsigned sw = to_floatbv_type(cast.type).fraction;

  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);
  const z3_smt_sort *zs = static_cast<const z3_smt_sort *>(s);

  // Convert each type
  if(is_bool_type(cast.from)) {
    // For bools, there is no direct conversion, so the cast is
    // transformed into fpa = b ? 1 : 0;
    expr2tc zero_expr;
    migrate_expr(gen_zero(migrate_type_back(cast.type)), zero_expr);

    expr2tc one_expr;
    migrate_expr(gen_one(migrate_type_back(cast.type)), one_expr);

    const smt_ast *args[3];
    args[0] = from;
    args[1] = convert_ast(one_expr);
    args[2] = convert_ast(zero_expr);

    return mk_func_app(s, SMT_FUNC_ITE, args, 3);
  } if(is_unsignedbv_type(cast.from)) {
    return new_ast(ctx.fpa_from_unsigned(mrm_const->e, mfrom->e, zs->s), s);
  } else if(is_signedbv_type(cast.from)) {
    return new_ast(ctx.fpa_from_signed(mrm_const->e, mfrom->e, zs->s), s);
  } else if(is_floatbv_type(cast.from)) {
    return new_ast(ctx.fpa_to_fpa(mrm_const->e, mfrom->e, zs->s), s);
  }

  abort();
}

smt_astt z3_convt::mk_smt_nearbyint_from_float(const nearbyint2t& expr)
{
  // Rounding mode symbol
  smt_astt rm = convert_rounding_mode(expr.rounding_mode);
  const z3_smt_ast *mrm = z3_smt_downcast(rm);

  smt_astt from = convert_ast(expr.from);
  const z3_smt_ast *mfrom = z3_smt_downcast(from);

  smt_sortt s = convert_sort(expr.type);
  return new_ast(ctx.fpa_to_integral(mrm->e, mfrom->e), s);
}

smt_astt z3_convt::mk_smt_bvfloat_arith_ops(const expr2tc& expr)
{
  // Rounding mode symbol
  smt_astt rm = convert_rounding_mode(*expr->get_sub_expr(2));
  const z3_smt_ast *mrm = z3_smt_downcast(rm);

  unsigned ew = to_floatbv_type(expr->type).exponent;
  unsigned sw = to_floatbv_type(expr->type).fraction;
  smt_sort *s = mk_sort(SMT_SORT_FLOATBV, ew, sw);

  // Sides
  smt_astt s1 = convert_ast(*expr->get_sub_expr(0));
  const z3_smt_ast *ms1 = z3_smt_downcast(s1);

  smt_astt s2 = convert_ast(*expr->get_sub_expr(1));
  const z3_smt_ast *ms2 = z3_smt_downcast(s2);

  switch (expr->expr_id) {
    case expr2t::ieee_add_id:
      return new_ast(ctx.fpa_add(mrm->e, ms1->e, ms2->e), s);
    case expr2t::ieee_sub_id:
      return new_ast(ctx.fpa_sub(mrm->e, ms1->e, ms2->e), s);
    case expr2t::ieee_mul_id:
      return new_ast(ctx.fpa_mul(mrm->e, ms1->e, ms2->e), s);
    case expr2t::ieee_div_id:
      return new_ast(ctx.fpa_div(mrm->e, ms1->e, ms2->e), s);
    case expr2t::ieee_fma_id:
    {
      smt_astt v3 = convert_ast(*expr->get_sub_expr(3));
      const z3_smt_ast *mv3 = z3_smt_downcast(v3);

      return new_ast(ctx.fpa_fma(mrm->e, ms1->e, ms2->e, mv3->e), s);
    }
    default:
      break;
  }

  abort();
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
  z3_smt_sort *s = NULL;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    s = new z3_smt_sort(k, ctx.int_sort(), 0);
    break;
  case SMT_SORT_REAL:
    s = new z3_smt_sort(k, ctx.real_sort());
    break;
  case SMT_SORT_BV:
  {
    unsigned long uint = va_arg(ap, unsigned long);
    s = new z3_smt_sort(k, ctx.bv_sort(uint), uint);
    break;
  }
  case SMT_SORT_ARRAY:
  {
    z3_smt_sort *dom = va_arg(ap, z3_smt_sort *); // Consider constness?
    z3_smt_sort *range = va_arg(ap, z3_smt_sort *);
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
  case SMT_SORT_FLOATBV:
  {
    unsigned ew = va_arg(ap, unsigned long);
    unsigned sw = va_arg(ap, unsigned long);

    // We need to add an extra bit to the significand size,
    // as it has no hidden bit
    s = new z3_smt_sort(k, ctx.fpa_sort(ew, sw + 1), ew + sw + 1);
    break;
  }
  case SMT_SORT_FLOATBV_RM:
    s = new z3_smt_sort(k, ctx.fpa_rm_sort());
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

const smt_ast *
z3_smt_ast::eq(smt_convt *ctx, const smt_ast *other) const
{
  const smt_sort *boolsort = ctx->mk_sort(SMT_SORT_BOOL);
  const smt_ast *args[2];
  args[0] = this;
  args[1] = other;
  return ctx->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
}

const smt_ast *
z3_smt_ast::update(smt_convt *conv, const smt_ast *value,
                   unsigned int idx, expr2tc idx_expr) const
{
  expr2tc index;

  if (sort->id == SMT_SORT_ARRAY)
  {
    return smt_ast::update(conv, value, idx, idx_expr);
  }
  else
  {
    assert(sort->id == SMT_SORT_STRUCT || sort->id == SMT_SORT_UNION);
    assert(is_nil_expr(idx_expr) &&
           "Can only update constant index tuple elems");

    z3_convt *z3_conv = static_cast<z3_convt*>(conv);
    const z3_smt_ast *updateval = z3_smt_downcast(value);
    return z3_conv->new_ast(z3_conv->mk_tuple_update(e, idx, updateval->e), sort);
  }
}

const smt_ast *
z3_smt_ast::select(smt_convt *ctx, const expr2tc &idx) const
{
  const smt_ast *args[2];
  args[0] = this;
  args[1] = ctx->convert_ast(idx);
  const smt_sort *rangesort = z3_sort_downcast(sort)->rangesort;
  return ctx->mk_func_app(rangesort, SMT_FUNC_SELECT, args, 2);
}

const smt_ast *
z3_smt_ast::project(smt_convt *conv, unsigned int elem) const
{
  z3_convt *z3_conv = static_cast<z3_convt*>(conv);

  const z3_smt_sort *thesort = z3_sort_downcast(sort);
  assert(!is_nil_type(thesort->tupletype));
  const struct_union_data &data = conv->get_type_def(thesort->tupletype);
  assert(elem < data.members.size());
  const smt_sort *idx_sort = conv->convert_sort(data.members[elem]);

  return z3_conv->new_ast(z3_conv->mk_tuple_select(e, elem), idx_sort);
}

smt_astt
z3_convt::tuple_create(const expr2tc &structdef)
{
  z3::expr e;
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type =
    static_cast<const struct_union_data &>(*strct.type);

  convert_struct(strct.datatype_members, type.members, strct.type, e);
  smt_sort *s = mk_struct_sort(structdef->type);
  return new_ast(e, s);
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
                                    .value.to_uint64();
    uint64_t offs = to_constant_int2t(outstruct->datatype_members[1])
                                     .value.to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  return outstruct;
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

  if (Z3_get_bool_value(ctx, e) == Z3_L_TRUE)
    return gen_true_expr();
  else
    return gen_false_expr();
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

  if (Z3_get_ast_kind(ctx, e) == Z3_NUMERAL_AST)
  {
    std::string value = Z3_get_numeral_string(ctx, e);
    return constant_int2tc(t, BigInt(value.c_str()));
  }
  else if (Z3_get_ast_kind(ctx, e) == Z3_APP_AST)
  {
    // floatbv?
    if(!is_floatbv_type(t))
      return expr2tc();

    unsigned ew = Z3_fpa_get_ebits(ctx, e.get_sort());

    // Remove an extra bit added when creating the sort,
    // because we represent the hidden bit like Z3 does
    unsigned sw = Z3_fpa_get_sbits(ctx, e.get_sort()) - 1;

    ieee_float_spect spec(sw, ew);
    ieee_floatt value(spec);

    Z3_ast v;
    if(Z3_model_eval(ctx, model, Z3_mk_fpa_to_ieee_bv(ctx, e), 1, &v))
    {
      ieee_floatt number(spec);
      number.unpack(BigInt(Z3_get_numeral_string(ctx, v)));

      return constant_floatbv2tc(t, number);
    }
  }

  return expr2tc();
}

expr2tc
z3_convt::get_array_elem(const smt_ast *array, uint64_t index,
                         const type2tc &subtype)
{
  const z3_smt_ast *za = z3_smt_downcast(array);
  unsigned long array_bound = array->sort->domain_width;
  const z3_smt_ast *idx;
  if (int_encoding)
    idx = static_cast<const z3_smt_ast*>(mk_smt_int(BigInt(index), false));
  else
    idx = static_cast<const z3_smt_ast*>(mk_smt_bvint(BigInt(index), false, array_bound));

  z3::expr e = select(za->e, idx->e);
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  z3_smt_ast *value = new_ast(e, convert_sort(subtype));
  unsigned long bv_size = array->sort->data_width;
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

  z3::expr e = z3::to_expr(ctx, Z3_mk_or(ctx, v.size(), arr));
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

  z3::expr e = z3::to_expr(ctx, Z3_mk_and(ctx, v.size(), arr));
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

void z3_smt_ast::dump() const
{
  std::cout << Z3_ast_to_string(e.ctx(), e) << std::endl;
  std::cout << "sort is " << Z3_sort_to_string(e.ctx(), Z3_get_sort(e.ctx(), e)) << std::endl;
}

void z3_convt::dump_smt()
{
  std::cout << solver << std::endl;
}
