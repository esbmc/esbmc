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
#define DEBUGLOC                                                               \
  std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

smt_convt *create_new_z3_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  z3_convt *conv = new z3_convt(int_encoding, ns);
  *tuple_api = static_cast<tuple_iface *>(conv);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

z3_convt::z3_convt(bool int_encoding, const namespacet &_ns)
  : smt_convt(int_encoding, _ns),
    array_iface(true, true),
    fp_convt(ctx),
    z3_ctx(false)
{
  z3::config conf;
  z3_ctx.init(conf, int_encoding);

  solver = (z3::tactic(z3_ctx, "simplify") & z3::tactic(z3_ctx, "solve-eqs") &
            z3::tactic(z3_ctx, "simplify") & z3::tactic(z3_ctx, "smt"))
             .mk_solver();

  z3::params p(z3_ctx);
  p.set("relevancy", (unsigned int)0);
  p.set("model", true);
  p.set("proof", false);
  solver.set(p);

  Z3_set_ast_print_mode(z3_ctx, Z3_PRINT_SMTLIB2_COMPLIANT);

  assumpt_ctx_stack.push_back(assumpt.begin());
}

z3_convt::~z3_convt()
{
}

void z3_convt::push_ctx()
{
  smt_convt::push_ctx();
  intr_push_ctx();
  solver.push();
}

void z3_convt::pop_ctx()
{
  solver.pop();
  intr_pop_ctx();
  smt_convt::pop_ctx();

  // Clear model if we have one.
  model = z3::model();
}

void z3_convt::intr_push_ctx()
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

void z3_convt::intr_pop_ctx()
{
  // Erase everything on stack since last push_ctx
  std::list<z3::expr>::iterator it = assumpt_ctx_stack.back();
  ++it;
  assumpt.erase(it, assumpt.end());
  assumpt_ctx_stack.pop_back();
}

smt_convt::resultt z3_convt::dec_solve()
{
  pre_solve();

  z3::check_result result = solver.check();

  if(result == z3::sat)
  {
    model = solver.get_model();
    return P_SATISFIABLE;
  }

  if(result == z3::unsat)
    return smt_convt::P_UNSATISFIABLE;

  return smt_convt::P_ERROR;
}

void z3_convt::assert_ast(const smt_ast *a)
{
  z3::expr theval = to_solver_smt_ast<z3_smt_ast>(a)->a;
  solver.add(theval);
  assumpt.push_back(theval);
}

z3::expr
z3_convt::mk_tuple_update(const z3::expr &t, unsigned i, const z3::expr &newval)
{
  z3::sort ty;
  unsigned num_fields, j;

  ty = t.get_sort();

  if(!ty.is_datatype())
  {
    std::cerr << "argument must be a tuple";
    abort();
  }

  num_fields = Z3_get_tuple_sort_num_fields(z3_ctx, ty);

  if(i >= num_fields)
  {
    std::cerr << "invalid tuple update, index is too big";
    abort();
  }

  std::vector<z3::expr> new_fields;
  new_fields.resize(num_fields);
  for(j = 0; j < num_fields; j++)
  {
    if(i == j)
    {
      /* use new_val at position i */
      new_fields[j] = newval;
    }
    else
    {
      /* use field j of t */
      z3::func_decl proj_decl =
        z3::to_func_decl(z3_ctx, Z3_get_tuple_sort_field_decl(z3_ctx, ty, j));
      new_fields[j] = proj_decl(t);
    }
  }

  z3::func_decl mk_tuple_decl =
    z3::to_func_decl(z3_ctx, Z3_get_tuple_sort_mk_decl(z3_ctx, ty));

  return mk_tuple_decl.make_tuple_from_array(num_fields, new_fields.data());
}

z3::expr z3_convt::mk_tuple_select(const z3::expr &t, unsigned i)
{
  z3::sort ty;
  unsigned num_fields;

  ty = t.get_sort();

  if(!ty.is_datatype())
  {
    std::cerr << "Z3 conversion: argument must be a tuple" << std::endl;
    abort();
  }

  num_fields = Z3_get_tuple_sort_num_fields(z3_ctx, ty);

  if(i >= num_fields)
  {
    std::cerr << "Z3 conversion: invalid tuple select, index is too large"
              << std::endl;
    abort();
  }

  z3::func_decl proj_decl =
    z3::to_func_decl(z3_ctx, Z3_get_tuple_sort_field_decl(z3_ctx, ty, i));
  return proj_decl(t);
}

// SMT-abstraction migration routines.

smt_astt z3_convt::mk_func_app(
  const smt_sort *s,
  smt_func_kind k,
  const smt_ast *const *args,
  unsigned int numargs)
{
  const z3_smt_ast *asts[4];
  unsigned int i;

  assert(numargs <= 4);
  for(i = 0; i < numargs; i++)
    asts[i] = to_solver_smt_ast<z3_smt_ast>(args[i]);

  switch(k)
  {
  case SMT_FUNC_ADD:
  case SMT_FUNC_BVADD:
    return new_ast((asts[0]->a + asts[1]->a), s);
  case SMT_FUNC_SUB:
  case SMT_FUNC_BVSUB:
    return new_ast((asts[0]->a - asts[1]->a), s);
  case SMT_FUNC_MUL:
  case SMT_FUNC_BVMUL:
    return new_ast((asts[0]->a * asts[1]->a), s);
  case SMT_FUNC_MOD:
    if(s->id == SMT_SORT_FLOATBV)
      return new_ast(
        z3::to_expr(z3_ctx, Z3_mk_fpa_rem(z3_ctx, asts[0]->a, asts[1]->a)), s);
    else
      return new_ast(
        z3::to_expr(z3_ctx, Z3_mk_mod(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVSMOD:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvsrem(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVUMOD:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvurem(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_DIV:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_div(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVSDIV:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvsdiv(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVUDIV:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvudiv(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_SHL:
    return new_ast(asts[0]->a * pw(z3_ctx.int_val(2), asts[1]->a), s);
  case SMT_FUNC_BVSHL:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvshl(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVASHR:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvashr(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_NEG:
  case SMT_FUNC_BVNEG:
    return new_ast((-asts[0]->a), s);
  case SMT_FUNC_BVLSHR:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_bvlshr(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_BVNOT:
    return new_ast((~asts[0]->a), s);
  case SMT_FUNC_BVNXOR:
    return new_ast(!(asts[0]->a ^ asts[1]->a), s);
  case SMT_FUNC_BVNOR:
    return new_ast(!(asts[0]->a | asts[1]->a), s);
  case SMT_FUNC_BVNAND:
    return new_ast(!(asts[0]->a & asts[1]->a), s);
  case SMT_FUNC_BVXOR:
    return new_ast((asts[0]->a ^ asts[1]->a), s);
  case SMT_FUNC_BVOR:
    return new_ast((asts[0]->a | asts[1]->a), s);
  case SMT_FUNC_BVAND:
    return new_ast((asts[0]->a & asts[1]->a), s);
  case SMT_FUNC_IMPLIES:
    return new_ast(implies(asts[0]->a, asts[1]->a), s);
  case SMT_FUNC_XOR:
    return new_ast(mk_xor(asts[0]->a, asts[1]->a), s);
  case SMT_FUNC_OR:
    return new_ast((asts[0]->a || asts[1]->a), s);
  case SMT_FUNC_AND:
    return new_ast((asts[0]->a && asts[1]->a), s);
  case SMT_FUNC_NOT:
    return new_ast(!asts[0]->a, s);
  // NB: mk_{l,g}t{,e} ignore unsigned arg in integer mode.
  case SMT_FUNC_LT:
  case SMT_FUNC_BVULT:
    return new_ast(mk_lt(asts[0]->a, asts[1]->a, true), s);
  case SMT_FUNC_BVSLT:
    return new_ast(mk_lt(asts[0]->a, asts[1]->a, false), s);
  case SMT_FUNC_GT:
  case SMT_FUNC_BVUGT:
    return new_ast(mk_gt(asts[0]->a, asts[1]->a, true), s);
  case SMT_FUNC_BVSGT:
    return new_ast(mk_gt(asts[0]->a, asts[1]->a, false), s);
  case SMT_FUNC_LTE:
  case SMT_FUNC_BVULTE:
    return new_ast(mk_le(asts[0]->a, asts[1]->a, true), s);
  case SMT_FUNC_BVSLTE:
    return new_ast(mk_le(asts[0]->a, asts[1]->a, false), s);
  case SMT_FUNC_GTE:
  case SMT_FUNC_BVUGTE:
    return new_ast(mk_ge(asts[0]->a, asts[1]->a, true), s);
  case SMT_FUNC_BVSGTE:
    return new_ast(mk_ge(asts[0]->a, asts[1]->a, false), s);
  case SMT_FUNC_EQ:
    return new_ast((asts[0]->a == asts[1]->a), s);
  case SMT_FUNC_NOTEQ:
    return new_ast((asts[0]->a != asts[1]->a), s);
  case SMT_FUNC_ITE:
    return new_ast(ite(asts[0]->a, asts[1]->a, asts[2]->a), s);
  case SMT_FUNC_STORE:
    return new_ast(store(asts[0]->a, asts[1]->a, asts[2]->a), s);
  case SMT_FUNC_SELECT:
    return new_ast(select(asts[0]->a, asts[1]->a), s);
  case SMT_FUNC_CONCAT:
    return new_ast(
      z3::to_expr(z3_ctx, Z3_mk_concat(z3_ctx, asts[0]->a, asts[1]->a)), s);
  case SMT_FUNC_REAL2INT:
    return new_ast(z3::to_expr(z3_ctx, Z3_mk_real2int(z3_ctx, asts[0]->a)), s);
  case SMT_FUNC_INT2REAL:
    return new_ast(z3::to_expr(z3_ctx, Z3_mk_int2real(z3_ctx, asts[0]->a)), s);
  case SMT_FUNC_IS_INT:
    return new_ast(z3::to_expr(z3_ctx, Z3_mk_is_int(z3_ctx, asts[0]->a)), s);
  case SMT_FUNC_BV2FLOAT:
    return new_ast(
      z3_ctx.fpa_from_bv(asts[0]->a, to_solver_smt_sort<z3::sort>(s)->s), s);
  case SMT_FUNC_FLOAT2BV:
    return new_ast(z3_ctx.fpa_to_ieeebv(asts[0]->a), s);
  default:
    std::cerr << "Unhandled SMT func in z3 conversion" << std::endl;
    abort();
  }
}

smt_astt z3_convt::mk_extract(
  const smt_ast *a,
  unsigned int high,
  unsigned int low,
  const smt_sort *s)
{
  // If it's a floatbv, convert it to bv
  if(a->sort->id == SMT_SORT_FLOATBV)
    a = new_ast(z3_ctx.fpa_to_ieeebv(to_solver_smt_ast<z3_smt_ast>(a)->a), s);

  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_extract(z3_ctx, high, low, to_solver_smt_ast<z3_smt_ast>(a)->a)),
    s);
}

smt_astt z3_convt::mk_smt_int(
  const mp_integer &theint,
  bool sign __attribute__((unused)))
{
  smt_sortt s = mk_int_sort();
  if(theint.is_negative())
    return new_ast(z3_ctx.int_val(theint.to_int64()), s);

  return new_ast(z3_ctx.int_val(theint.to_uint64()), s);
}

smt_astt z3_convt::mk_smt_real(const std::string &str)
{
  smt_sortt s = mk_real_sort();
  return new_ast(z3_ctx.real_val(str.c_str()), s);
}

smt_astt z3_convt::mk_smt_bv(smt_sortt s, const mp_integer &theint)
{
  std::size_t w = s->get_data_width();

  if(theint.is_negative())
    return new_ast(z3_ctx.bv_val(theint.to_int64(), w), s);

  return new_ast(z3_ctx.bv_val(theint.to_uint64(), w), s);
}

smt_astt z3_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  smt_sortt s = mk_real_fp_sort(thereal.spec.e, thereal.spec.f);

  const mp_integer sig = thereal.get_fraction();

  // If the number is denormal, we set the exponent to -bias
  const mp_integer exp =
    thereal.is_normal() ? thereal.get_exponent() + thereal.spec.bias() : 0;

  smt_astt sgn_bv = ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(thereal.get_sign()), 1);
  smt_astt exp_bv = ctx->mk_smt_bv(SMT_SORT_UBV, exp, thereal.spec.e);
  smt_astt sig_bv = ctx->mk_smt_bv(SMT_SORT_UBV, sig, thereal.spec.f);

  return new_ast(
    z3_ctx.fpa_val(
      to_solver_smt_ast<z3_smt_ast>(sgn_bv)->a,
      to_solver_smt_ast<z3_smt_ast>(exp_bv)->a,
      to_solver_smt_ast<z3_smt_ast>(sig_bv)->a),
    s);
}

smt_astt z3_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw);
  return new_ast(z3_ctx.fpa_nan(to_solver_smt_sort<z3::sort>(s)->s), s);
}

smt_astt z3_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  smt_sortt s = mk_real_fp_sort(ew, sw);
  return new_ast(z3_ctx.fpa_inf(sgn, to_solver_smt_sort<z3::sort>(s)->s), s);
}

smt_astt z3_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  smt_sortt s = fp_api->mk_fpbv_rm_sort();

  switch(rm)
  {
  case ieee_floatt::ROUND_TO_EVEN:
    return new_ast(z3_ctx.fpa_rm_ne(), s);
  case ieee_floatt::ROUND_TO_MINUS_INF:
    return new_ast(z3_ctx.fpa_rm_mi(), s);
  case ieee_floatt::ROUND_TO_PLUS_INF:
    return new_ast(z3_ctx.fpa_rm_pi(), s);
  case ieee_floatt::ROUND_TO_ZERO:
    return new_ast(z3_ctx.fpa_rm_ze(), s);
  default:
    break;
  }

  abort();
}

smt_astt z3_convt::mk_smt_bool(bool val)
{
  return new_ast(z3_ctx.bool_val(val), boolean_sort);
}

smt_astt z3_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_astt z3_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  return new_ast(
    z3_ctx.constant(name.c_str(), to_solver_smt_sort<z3::sort>(s)->s), s);
}

smt_sortt z3_convt::mk_struct_sort(const type2tc &type)
{
  if(is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    smt_sortt subtypesort = convert_sort(arrtype.subtype);
    smt_sortt d = mk_int_bv_sort(
      SMT_SORT_UBV, make_array_domain_type(arrtype)->get_width());
    return mk_array_sort(d, subtypesort);
  }

  const struct_type2t &strct = to_struct_type(type);
  const std::vector<type2tc> &members = strct.members;
  const std::vector<irep_idt> &member_names = strct.member_names;
  const irep_idt &struct_name = strct.name;

  z3::symbol mk_tuple_name, *proj_names;
  z3::sort *proj_types;
  Z3_func_decl mk_tuple_decl, *proj_decls;
  std::string name;
  u_int num_elems = members.size();

  proj_names = new z3::symbol[num_elems];
  proj_types = new z3::sort[num_elems];
  proj_decls = new Z3_func_decl[num_elems];

  name = "struct";
  name += "_type_" + struct_name.as_string();
  mk_tuple_name = z3::symbol(z3_ctx, name.c_str());

  z3::sort sort;
  if(!members.size())
  {
    sort = z3::to_sort(
      z3_ctx,
      Z3_mk_tuple_sort(
        z3_ctx, mk_tuple_name, 0, nullptr, nullptr, &mk_tuple_decl, nullptr));
  }
  else
  {
    u_int i = 0;
    std::vector<irep_idt>::const_iterator mname = member_names.begin();
    for(std::vector<type2tc>::const_iterator it = members.begin();
        it != members.end();
        it++, mname++, i++)
    {
      proj_names[i] = z3::symbol(z3_ctx, mname->as_string().c_str());
      auto tmp = to_solver_smt_sort<z3::sort>(convert_sort(*it));
      proj_types[i] = tmp->s;
    }

    // Unpack pointers from Z3++ objects.
    Z3_symbol *unpacked_symbols = new Z3_symbol[num_elems];
    Z3_sort *unpacked_sorts = new Z3_sort[num_elems];
    for(i = 0; i < num_elems; i++)
    {
      unpacked_symbols[i] = proj_names[i];
      unpacked_sorts[i] = proj_types[i];
    }

    sort = z3::to_sort(
      z3_ctx,
      Z3_mk_tuple_sort(
        z3_ctx,
        mk_tuple_name,
        num_elems,
        unpacked_symbols,
        unpacked_sorts,
        &mk_tuple_decl,
        proj_decls));

    delete[] unpacked_symbols;
    delete[] unpacked_sorts;
    delete[] proj_names;
    delete[] proj_types;
    delete[] proj_decls;
  }

  return new solver_smt_sort<z3::sort>(SMT_SORT_STRUCT, sort, type);
}

const smt_ast *z3_smt_ast::update(
  smt_convt *conv,
  const smt_ast *value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  if(sort->id == SMT_SORT_ARRAY)
    return smt_ast::update(conv, value, idx, idx_expr);

  assert(sort->id == SMT_SORT_STRUCT);
  assert(is_nil_expr(idx_expr) && "Can only update constant index tuple elems");

  z3_convt *z3_conv = static_cast<z3_convt *>(conv);
  const z3_smt_ast *updateval = to_solver_smt_ast<z3_smt_ast>(value);
  return z3_conv->new_ast(z3_conv->mk_tuple_update(a, idx, updateval->a), sort);
}

const smt_ast *z3_smt_ast::project(smt_convt *conv, unsigned int elem) const
{
  z3_convt *z3_conv = static_cast<z3_convt *>(conv);

  assert(!is_nil_type(sort->get_tuple_type()));
  const struct_union_data &data = conv->get_type_def(sort->get_tuple_type());

  assert(elem < data.members.size());
  const smt_sort *idx_sort = conv->convert_sort(data.members[elem]);

  return z3_conv->new_ast(z3_conv->mk_tuple_select(a, elem), idx_sort);
}

smt_astt z3_convt::tuple_create(const expr2tc &structdef)
{
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type =
    static_cast<const struct_union_data &>(*strct.type);

  // Converts a static struct - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  const std::vector<expr2tc> &members = strct.datatype_members;
  const std::vector<type2tc> &member_types = type.members;

  unsigned size = member_types.size();

  z3::expr *args = new z3::expr[size];

#ifndef NDEBUG
  unsigned int numoperands = members.size();
  assert(
    numoperands == member_types.size() &&
    "Too many / few struct fields for struct type");
#endif

  // Populate tuple with members of that struct
  for(unsigned int i = 0; i < member_types.size(); i++)
  {
    const z3_smt_ast *tmp =
      to_solver_smt_ast<z3_smt_ast>(convert_ast(members[i]));
    args[i] = tmp->a;
  }

  // Create tuple itself, return to caller. This is a lump of data, we don't
  // need to bind it to a name or symbol.
  smt_sortt s = mk_struct_sort(structdef->type);

  Z3_func_decl decl =
    Z3_get_tuple_sort_mk_decl(z3_ctx, to_solver_smt_sort<z3::sort>(s)->s);
  z3::func_decl d(z3_ctx, decl);
  z3::expr e = d.make_tuple_from_array(size, args);
  delete[] args;

  return new_ast(e, s);
}

smt_astt z3_convt::tuple_fresh(const smt_sort *s, std::string name)
{
  const char *n = (name == "") ? nullptr : name.c_str();
  z3::expr output = z3_ctx.fresh_const(n, to_solver_smt_sort<z3::sort>(s)->s);
  return new_ast(output, s);
}

const smt_ast *z3_convt::convert_array_of(
  smt_astt init_val,
  unsigned long domain_width)
{
  smt_sortt dom_sort = mk_int_bv_sort(SMT_SORT_UBV, domain_width);

  z3::expr val = to_solver_smt_ast<z3_smt_ast>(init_val)->a;
  z3::expr output = z3::to_expr(
    z3_ctx,
    Z3_mk_const_array(z3_ctx, to_solver_smt_sort<z3::sort>(dom_sort)->s, val));

  return new_ast(output, mk_array_sort(dom_sort, init_val->sort));
}

const smt_ast *z3_convt::tuple_array_create(
  const type2tc &arr_type,
  const smt_ast **input_args,
  bool const_array,
  const smt_sort *domain)
{
  z3::expr output;
  const array_type2t &arrtype = to_array_type(arr_type);

  if(const_array)
  {
    z3::expr value, index;
    z3::sort array_type, dom_type;
    std::string tmp, identifier;

    array_type = to_solver_smt_sort<z3::sort>(convert_sort(arr_type))->s;
    dom_type = array_type.array_domain();

    const z3_smt_ast *tmpast = to_solver_smt_ast<z3_smt_ast>(*input_args);
    value = tmpast->a;

    if(is_bool_type(arrtype.subtype))
    {
      value = z3_ctx.bool_val(false);
    }

    output = z3::to_expr(z3_ctx, Z3_mk_const_array(z3_ctx, dom_type, value));
  }
  else
  {
    u_int i = 0;
    z3::sort z3_array_type;
    z3::expr int_cte, val_cte;
    z3::sort domain_sort;

    assert(
      !is_nil_expr(arrtype.array_size) &&
      "Non-const array-of's can't be infinitely sized");
    const constant_int2t &sz = to_constant_int2t(arrtype.array_size);

    assert(
      is_constant_int2t(arrtype.array_size) &&
      "array_of sizes should be constant");

    int64_t size;
    size = sz.as_long();

    z3_array_type = to_solver_smt_sort<z3::sort>(convert_sort(arr_type))->s;
    domain_sort = z3_array_type.array_domain();

    output = z3_ctx.fresh_const(nullptr, z3_array_type);

    for(i = 0; i < size; i++)
    {
      int_cte = z3_ctx.num_val(i, domain_sort);
      const z3_smt_ast *tmpast = to_solver_smt_ast<z3_smt_ast>(input_args[i]);
      output = z3::store(output, int_cte, tmpast->a);
    }
  }

  smt_sortt ssort = mk_struct_sort(arrtype.subtype);
  smt_sortt asort = mk_array_sort(domain, ssort);
  return new_ast(output, asort);
}

smt_astt z3_convt::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_astt z3_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
}

smt_astt z3_convt::tuple_array_of(
  const expr2tc &init,
  unsigned long domain_width)
{
  return convert_array_of(convert_ast(init), domain_width);
}

expr2tc z3_convt::tuple_get(const expr2tc &expr)
{
  if(is_pointer_type(expr->type))
    return expr2tc();

  const struct_union_data &strct = get_type_def(expr->type);

  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  for(auto const &it : strct.members)
  {
    member2tc memb(it, expr, strct.member_names[i]);
    outstruct->datatype_members.push_back(get(memb));
    i++;
  }

  // If it's a pointer, rewrite.
  if(is_pointer_type(expr->type))
  {
    uint64_t num =
      to_constant_int2t(outstruct->datatype_members[0]).value.to_uint64();
    uint64_t offs =
      to_constant_int2t(outstruct->datatype_members[1]).value.to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  return outstruct;
}

// ***************************** 'get' api *******************************

expr2tc z3_convt::get_bool(const smt_ast *a)
{
  assert(a->sort->id == SMT_SORT_BOOL);

  const z3_smt_ast *za = to_solver_smt_ast<z3_smt_ast>(a);
  z3::expr e = model.eval(za->a, false);

  if(Z3_get_bool_value(z3_ctx, e) == Z3_L_TRUE)
    return gen_true_expr();

  return gen_false_expr();
}

expr2tc z3_convt::get_bv(const type2tc &type, smt_astt a)
{
  assert(a->sort->id >= SMT_SORT_SBV || a->sort->id <= SMT_SORT_FIXEDBV);

  const z3_smt_ast *za = to_solver_smt_ast<z3_smt_ast>(a);
  z3::expr e = model.eval(za->a, false);

  // Not a numeral? Let's not try to convert it
  if(Z3_get_ast_kind(z3_ctx, e) != Z3_NUMERAL_AST)
    return expr2tc();

  BigInt val = string2integer(Z3_get_numeral_string(z3_ctx, e));
  return build_bv(type, val);
}

ieee_floatt z3_convt::get_fpbv(smt_astt a)
{
  const z3_smt_ast *za = to_solver_smt_ast<z3_smt_ast>(a);
  z3::expr e = model.eval(za->a, false);

  assert(Z3_get_ast_kind(z3_ctx, e) == Z3_APP_AST);

  unsigned ew = Z3_fpa_get_ebits(z3_ctx, e.get_sort());

  // Remove an extra bit added when creating the sort,
  // because we represent the hidden bit like Z3 does
  unsigned sw = Z3_fpa_get_sbits(z3_ctx, e.get_sort()) - 1;

  ieee_floatt number(ieee_float_spect(sw, ew));
  number.make_zero();

  if(Z3_fpa_is_numeral_nan(z3_ctx, e))
    number.make_NaN();
  else if(Z3_fpa_is_numeral_inf(z3_ctx, e))
  {
    if(Z3_fpa_is_numeral_positive(z3_ctx, e))
      number.make_plus_infinity();
    else
      number.make_minus_infinity();
  }
  else
  {
    Z3_ast v;
    if(Z3_model_eval(z3_ctx, model, Z3_mk_fpa_to_ieee_bv(z3_ctx, e), 1, &v))
      number.unpack(BigInt(Z3_get_numeral_string(z3_ctx, v)));
  }

  return number;
}

expr2tc z3_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  const z3_smt_ast *za = to_solver_smt_ast<z3_smt_ast>(array);
  unsigned long array_bound = array->sort->get_domain_width();
  const z3_smt_ast *idx;
  if(int_encoding)
    idx = to_solver_smt_ast<z3_smt_ast>(mk_smt_int(BigInt(index), false));
  else
    idx = to_solver_smt_ast<z3_smt_ast>(
      ctx->mk_smt_bv(SMT_SORT_UBV, BigInt(index), array_bound));

  z3::expr e = model.eval(select(za->a, idx->a), false);

  z3_smt_ast *value = new_ast(e, convert_sort(subtype));
  return get_by_ast(subtype, value);
}

const smt_ast *z3_convt::make_disjunct(const ast_vec &v)
{
  // Make a gigantic 'or'.
  Z3_ast arr[v.size()];

  size_t i = 0;
  for(auto const it : v)
    arr[i++] = to_solver_smt_ast<z3_smt_ast>(it)->a.operator Z3_ast();

  z3::expr e = z3::to_expr(z3_ctx, Z3_mk_or(z3_ctx, v.size(), arr));
  return new_ast(e, boolean_sort);
}

const smt_ast *z3_convt::make_conjunct(const ast_vec &v)
{
  // Make a gigantic 'and'.
  Z3_ast arr[v.size()];

  size_t i = 0;
  for(auto const it : v)
    arr[i++] = to_solver_smt_ast<z3_smt_ast>(it)->a.operator Z3_ast();

  z3::expr e = z3::to_expr(z3_ctx, Z3_mk_and(z3_ctx, v.size(), arr));
  return new_ast(e, boolean_sort);
}

void z3_convt::add_array_constraints_for_solving()
{
}

void z3_convt::push_array_ctx()
{
}

void z3_convt::pop_array_ctx()
{
}

void z3_convt::add_tuple_constraints_for_solving()
{
}

void z3_convt::push_tuple_ctx()
{
}

void z3_convt::pop_tuple_ctx()
{
}

void z3_smt_ast::dump() const
{
  std::cout << Z3_ast_to_string(a.ctx(), a) << std::endl;
  std::cout << "sort is " << Z3_sort_to_string(a.ctx(), Z3_get_sort(a.ctx(), a))
            << std::endl;
}

void z3_convt::dump_smt()
{
  std::cout << solver << std::endl;
}

void z3_convt::print_model()
{
  std::cout << Z3_model_to_string(z3_ctx, model);
}

smt_sortt z3_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  // We need to add an extra bit to the significand size,
  // as it has no hidden bit
  return new solver_smt_sort<z3::sort>(
    SMT_SORT_FLOATBV, z3_ctx.fpa_sort(ew, sw + 1), ew + sw + 1, sw);
}

smt_sortt z3_convt::mk_bv_fp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<z3::sort>(
    SMT_SORT_FAKE_FLOATBV, z3_ctx.bv_sort(ew + sw + 1), ew + sw + 1, sw + 1);
}

smt_sortt z3_convt::mk_fpbv_rm_sort()
{
  auto t = z3_ctx.fpa_rm_sort();
  return new solver_smt_sort<z3::sort>(SMT_SORT_FLOATBV_RM, t, 1);
}

smt_sortt z3_convt::mk_bool_sort()
{
  return new solver_smt_sort<z3::sort>(SMT_SORT_BOOL, z3_ctx.bool_sort(), 1);
}

smt_sortt z3_convt::mk_real_sort()
{
  return new solver_smt_sort<z3::sort>(SMT_SORT_REAL, z3_ctx.real_sort());
}

smt_sortt z3_convt::mk_int_sort()
{
  return new solver_smt_sort<z3::sort>(SMT_SORT_INT, z3_ctx.int_sort());
}

smt_sortt z3_convt::mk_bv_sort(const smt_sort_kind k, std::size_t width)
{
  return new solver_smt_sort<z3::sort>(k, z3_ctx.bv_sort(width), width);
}

smt_sortt z3_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<z3::sort>(domain);
  auto range_sort = to_solver_smt_sort<z3::sort>(range);

  auto t = z3_ctx.array_sort(domain_sort->s, range_sort->s);
  return new solver_smt_sort<z3::sort>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_astt z3_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, smt_sortt to)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const z3_smt_ast *mrm =
    to_solver_smt_ast<z3_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);
  return new_ast(z3_ctx.fpa_to_ubv(mrm->a, mfrom->a, to->get_data_width()), to);
}

smt_astt z3_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, smt_sortt to)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const z3_smt_ast *mrm =
    to_solver_smt_ast<z3_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);

  return new_ast(z3_ctx.fpa_to_sbv(mrm->a, mfrom->a, to->get_data_width()), to);
}

smt_astt z3_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);

  return new_ast(
    z3_ctx.fpa_to_fpa(mrm->a, mfrom->a, to_solver_smt_sort<z3::sort>(to)->s),
    to);
}

smt_astt
z3_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);

  return new_ast(
    z3_ctx.fpa_from_unsigned(
      mrm->a, mfrom->a, to_solver_smt_sort<z3::sort>(to)->s),
    to);
}

smt_astt
z3_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);

  return new_ast(
    z3_ctx.fpa_from_signed(
      mrm->a, mfrom->a, to_solver_smt_sort<z3::sort>(to)->s),
    to);
}

smt_astt z3_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mfrom = to_solver_smt_ast<z3_smt_ast>(from);
  return new_ast(z3_ctx.fpa_to_integral(mrm->a, mfrom->a), from->sort);
}

smt_astt z3_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mrd = to_solver_smt_ast<z3_smt_ast>(rd);
  return new_ast(z3_ctx.fpa_sqrt(mrm->a, mrd->a), rd->sort);
}

smt_astt
z3_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  const z3_smt_ast *mv1 = to_solver_smt_ast<z3_smt_ast>(v1);
  const z3_smt_ast *mv2 = to_solver_smt_ast<z3_smt_ast>(v2);
  const z3_smt_ast *mv3 = to_solver_smt_ast<z3_smt_ast>(v3);
  return new_ast(z3_ctx.fpa_fma(mrm->a, mv1->a, mv2->a, mv3->a), v1->sort);
}

smt_astt z3_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const z3_smt_ast *mlhs = to_solver_smt_ast<z3_smt_ast>(lhs);
  const z3_smt_ast *mrhs = to_solver_smt_ast<z3_smt_ast>(rhs);
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  return new_ast(z3_ctx.fpa_add(mrm->a, mlhs->a, mrhs->a), lhs->sort);
}

smt_astt z3_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const z3_smt_ast *mlhs = to_solver_smt_ast<z3_smt_ast>(lhs);
  const z3_smt_ast *mrhs = to_solver_smt_ast<z3_smt_ast>(rhs);
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  return new_ast(z3_ctx.fpa_sub(mrm->a, mlhs->a, mrhs->a), lhs->sort);
}

smt_astt z3_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const z3_smt_ast *mlhs = to_solver_smt_ast<z3_smt_ast>(lhs);
  const z3_smt_ast *mrhs = to_solver_smt_ast<z3_smt_ast>(rhs);
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  return new_ast(z3_ctx.fpa_mul(mrm->a, mlhs->a, mrhs->a), lhs->sort);
}

smt_astt z3_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const z3_smt_ast *mlhs = to_solver_smt_ast<z3_smt_ast>(lhs);
  const z3_smt_ast *mrhs = to_solver_smt_ast<z3_smt_ast>(rhs);
  const z3_smt_ast *mrm = to_solver_smt_ast<z3_smt_ast>(rm);
  return new_ast(z3_ctx.fpa_div(mrm->a, mlhs->a, mrhs->a), lhs->sort);
}

smt_astt z3_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_fpa_eq(
        z3_ctx,
        to_solver_smt_ast<z3_smt_ast>(lhs)->a,
        to_solver_smt_ast<z3_smt_ast>(rhs)->a)),
    lhs->sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx, Z3_mk_fpa_is_nan(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_fpa_is_infinite(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_fpa_is_normal(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx, Z3_mk_fpa_is_zero(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_fpa_is_negative(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx,
      Z3_mk_fpa_is_positive(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt z3_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    z3::to_expr(
      z3_ctx, Z3_mk_fpa_abs(z3_ctx, to_solver_smt_ast<z3_smt_ast>(op)->a)),
    op->sort);
}
