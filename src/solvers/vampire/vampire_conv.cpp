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
#include <vampire_conv.h>

#define new_ast new_solver_ast<vampire_smt_ast>

smt_convt *create_new_vampire_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  vampire_convt *conv = new vampire_convt(int_encoding, ns);
  *tuple_api = static_cast<tuple_iface *>(conv);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

vampire_convt::vampire_convt(bool int_encoding, const namespacet &_ns)
  : smt_convt(int_encoding, _ns),
    array_iface(true, true),
    solver(Vampire::Solver::getSolverPtr(Vampire::Solver::Logic::SMT_LIB)),
    fp_convt(this)
{}

void vampire_convt::push_ctx()
{
  //smt_convt::push_ctx();
  //solver.push();
}

void vampire_convt::pop_ctx()
{
  //solver.pop();
  //smt_convt::pop_ctx();
}

smt_convt::resultt vampire_convt::dec_solve()
{
  pre_solve();

  Vampire::Result result = solver->solve();

  if(result.satisfiable())
    return P_SATISFIABLE;

  if(result.unsatisfiable())
    return smt_convt::P_UNSATISFIABLE;

  return smt_convt::P_ERROR;
}

void vampire_convt::assert_ast(smt_astt a)
{
  Vampire::Expression exp = to_solver_smt_ast<vampire_smt_ast>(a)->a;
  solver->addFormula(exp);
}

Vampire::Expression
vampire_convt::mk_tuple_update(const Vampire::Expression &t, unsigned i, const Vampire::Expression &newval)
{
  
  /*Vampire::Sort sort = t.getSort();
  if(!sort.isTupleSort())
  {
    std::cerr << "argument must be a tuple";
    abort();
  }

  unsigned num_fields = t.arity();
  if(i >= num_fields)
  {
    std::cerr << "invalid tuple update, index is too big";
    abort();
  }

  std::vector<Vampire::Expression> args;
  for(unsigned j = 0; j < num_fields; j++)
  {
    if(i == j)
    {
      /* use new_val at position i */
    /*  args.push_back(newval);
    }
    else
    {
      /* use field j of t */
      /*vampire::func_decl proj_decl =
        vampire::to_func_decl(vampire_ctx, Z3_get_tuple_sort_field_decl(vampire_ctx, ty, j));
      args.push_back(proj_decl(t));
    }
  }

  return vampire::to_func_decl(vampire_ctx, Z3_get_tuple_sort_mk_decl(vampire_ctx, ty))(args);*/
  return solver->term(solver->function("f", 0));  
}

Vampire::Expression vampire_convt::mk_tuple_select(const Vampire::Expression &t, unsigned i)
{
  /*vampire::sort ty = t.get_sort();
  if(!ty.is_datatype())
  {
    std::cerr << "Z3 conversion: argument must be a tuple" << std::endl;
    abort();
  }

  size_t num_fields = Z3_get_tuple_sort_num_fields(vampire_ctx, ty);
  if(i >= num_fields)
  {
    std::cerr << "Z3 conversion: invalid tuple select, index is too large"
              << std::endl;
    abort();
  }

  vampire::func_decl proj_decl =
    vampire::to_func_decl(vampire_ctx, Z3_get_tuple_sort_field_decl(vampire_ctx, ty, i));
  return proj_decl(t);*/
  return solver->term(solver->function("f", 0));  
}

// SMT-abstraction migration routines.
smt_astt vampire_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    (solver->sum(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    (solver->difference(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}


smt_astt vampire_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    (solver->multiply(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}


smt_astt vampire_convt::mk_mod(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    (solver->mod(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}


smt_astt vampire_convt::mk_div(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  return new_ast(
    (solver->div(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_shl(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  std::cerr << "Vampire does not support the shift left operation" << std::endl;
  abort();
}


smt_astt vampire_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->neg(
      to_solver_smt_ast<vampire_smt_ast>(a)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    (solver->implies(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    (solver->exor(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    (solver->orFormula(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return new_ast(
    (solver->andFormula(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return new_ast(
    (solver->negation(
      to_solver_smt_ast<vampire_smt_ast>(a)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->lt(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->gt(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->leq(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->geq(
      to_solver_smt_ast<vampire_smt_ast>(a)->a,
      to_solver_smt_ast<vampire_smt_ast>(b)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  return a;
}

smt_astt vampire_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  return a;
}

smt_astt vampire_convt::mk_real2int(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->real2int(
      to_solver_smt_ast<vampire_smt_ast>(a)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_int2real(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return new_ast(
    (solver->int2real(
      to_solver_smt_ast<vampire_smt_ast>(a)->a)
    ),
    a->sort);
}

smt_astt vampire_convt::mk_isint(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  return a;
}

smt_astt vampire_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  // If it's a floatbv, convert it to bv
  return a;
}

smt_astt vampire_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  return a;
}

smt_astt vampire_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  return a;
}

smt_astt vampire_convt::mk_concat(smt_astt a, smt_astt b)
{
  return a;
}

smt_astt vampire_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  return cond;
}

smt_astt vampire_convt::mk_smt_int(const BigInt &theint)
{
  smt_sortt s = mk_int_sort();
  return new_ast(solver->integerConstant(theint.to_int64()), s);
}

smt_astt vampire_convt::mk_smt_real(const std::string &str)
{
  std::size_t found = str.find("/");
  if(found == std::string::npos){
    std::cerr << "Can only form a real from a string of the form a/b" << std::endl;
    abort();    
  }

  std::string num = str.substr(0, found - 1);
  std::string denom = str.substr(0, found + 1);

  smt_sortt s = mk_real_sort();
  return new_ast(solver->rationalConstant(num, denom), s);
}

smt_astt vampire_convt::mk_smt_bool(bool val)
{
  return new_ast(solver->boolFormula(val), boolean_sort);
}

/*smt_astt vampire_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype [[gnu::unused]])
{
  return mk_smt_symbol(name, s);
}

smt_astt vampire_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  return new_ast(
    vampire_ctx.constant(name.c_str(), to_solver_smt_sort<vampire::sort>(s)->s), s);
}

smt_sortt vampire_convt::mk_struct_sort(const type2tc &type)
{
  if(is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    smt_sortt subtypesort = convert_sort(arrtype.subtype);
    smt_sortt d = mk_int_bv_sort(make_array_domain_type(arrtype)->get_width());
    return mk_array_sort(d, subtypesort);
  }

  const struct_type2t &strct = to_struct_type(type);
  const std::size_t num_members = strct.members.size();

  vampire::array<Z3_symbol> member_names(num_members);
  vampire::array<Z3_sort> member_sorts(num_members);
  for(std::size_t i = 0; i < num_members; ++i)
  {
    member_names[i] =
      vampire_ctx.str_symbol(strct.member_names[i].as_string().c_str());
    member_sorts[i] =
      to_solver_smt_sort<vampire::sort>(convert_sort(strct.members[i]))->s;
  }

  vampire::symbol tuple_name = vampire_ctx.str_symbol(
    std::string("struct_type_" + strct.name.as_string()).c_str());

  Z3_func_decl mk_tuple_decl;
  vampire::array<Z3_func_decl> proj_decls(num_members);
  vampire::sort sort = to_sort(
    vampire_ctx,
    Z3_mk_tuple_sort(
      vampire_ctx,
      tuple_name,
      num_members,
      member_names.ptr(),
      member_sorts.ptr(),
      &mk_tuple_decl,
      proj_decls.ptr()));

  return new solver_smt_sort<vampire::sort>(SMT_SORT_STRUCT, sort, type);
}

smt_astt vampire_smt_ast::update(
  smt_convt *conv,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  if(sort->id == SMT_SORT_ARRAY)
    return smt_ast::update(conv, value, idx, idx_expr);

  assert(sort->id == SMT_SORT_STRUCT);
  assert(is_nil_expr(idx_expr) && "Can only update constant index tuple elems");

  vampire_convt *vampire_conv = static_cast<vampire_convt *>(conv);
  const vampire_smt_ast *updateval = to_solver_smt_ast<vampire_smt_ast>(value);
  return vampire_conv->new_ast(vampire_conv->mk_tuple_update(a, idx, updateval->a), sort);
}

smt_astt vampire_smt_ast::project(smt_convt *conv, unsigned int elem) const
{
  vampire_convt *vampire_conv = static_cast<vampire_convt *>(conv);

  assert(!is_nil_type(sort->get_tuple_type()));
  const struct_union_data &data = conv->get_type_def(sort->get_tuple_type());

  assert(elem < data.members.size());
  const smt_sort *idx_sort = conv->convert_sort(data.members[elem]);

  return vampire_conv->new_ast(vampire_conv->mk_tuple_select(a, elem), idx_sort);
}

smt_astt vampire_convt::tuple_create(const expr2tc &structdef)
{
  const constant_struct2t &strct = to_constant_struct2t(structdef);
  const struct_union_data &type =
    static_cast<const struct_union_data &>(*strct.type);

  // Converts a static struct - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  const std::vector<expr2tc> &members = strct.datatype_members;
  const std::vector<type2tc> &member_types = type.members;

  // Populate tuple with members of that struct
  vampire::expr_vector args(vampire_ctx);
  for(std::size_t i = 0; i < member_types.size(); ++i)
    args.push_back(to_solver_smt_ast<vampire_smt_ast>(convert_ast(members[i]))->a);

  // Create tuple itself, return to caller. This is a lump of data, we don't
  // need to bind it to a name or symbol.
  smt_sortt s = mk_struct_sort(structdef->type);

  vampire::func_decl vampire_tuple = vampire::to_func_decl(
    vampire_ctx,
    Z3_get_tuple_sort_mk_decl(vampire_ctx, to_solver_smt_sort<vampire::sort>(s)->s));
  return new_ast(vampire_tuple(args), s);
}

smt_astt vampire_convt::tuple_fresh(const smt_sort *s, std::string name)
{
  const char *n = (name == "") ? nullptr : name.c_str();
  return new_ast(vampire_ctx.constant(n, to_solver_smt_sort<vampire::sort>(s)->s), s);
}

smt_astt
vampire_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  smt_sortt dom_sort = mk_int_bv_sort(domain_width);

  vampire::expr val = to_solver_smt_ast<vampire_smt_ast>(init_val)->a;
  vampire::expr output = vampire::to_expr(
    vampire_ctx,
    Z3_mk_const_array(vampire_ctx, to_solver_smt_sort<vampire::sort>(dom_sort)->s, val));

  return new_ast(output, mk_array_sort(dom_sort, init_val->sort));
}

smt_astt vampire_convt::tuple_array_create(
  const type2tc &arr_type,
  smt_astt *input_args,
  bool const_array,
  const smt_sort *domain)
{
  const array_type2t &arrtype = to_array_type(arr_type);
  vampire::sort array_sort = to_solver_smt_sort<vampire::sort>(convert_sort(arr_type))->s;
  vampire::sort dom_sort = array_sort.array_domain();

  smt_sortt ssort = mk_struct_sort(arrtype.subtype);
  smt_sortt asort = mk_array_sort(domain, ssort);

  if(const_array)
  {
    const vampire_smt_ast *tmpast = to_solver_smt_ast<vampire_smt_ast>(*input_args);
    vampire::expr value = tmpast->a;

    if(is_bool_type(arrtype.subtype))
      value = vampire_ctx.bool_val(false);

    return new_ast(
      vampire::to_expr(vampire_ctx, Z3_mk_const_array(vampire_ctx, dom_sort, value)), asort);
  }

  assert(
    !is_nil_expr(arrtype.array_size) &&
    "Non-const array-of's can't be infinitely sized");

  assert(
    is_constant_int2t(arrtype.array_size) &&
    "array_of sizes should be constant");

  vampire::expr output = vampire_ctx.constant(nullptr, array_sort);
  for(std::size_t i = 0; i < to_constant_int2t(arrtype.array_size).as_ulong();
      ++i)
  {
    vampire::expr int_cte = vampire_ctx.num_val(i, dom_sort);
    const vampire_smt_ast *tmpast = to_solver_smt_ast<vampire_smt_ast>(input_args[i]);
    output = vampire::store(output, int_cte, tmpast->a);
  }

  return new_ast(output, asort);
}

smt_astt vampire_convt::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_astt vampire_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return mk_smt_symbol(sym.get_symbol_name(), convert_sort(sym.type));
}

smt_astt
vampire_convt::tuple_array_of(const expr2tc &init, unsigned long domain_width)
{
  return convert_array_of(convert_ast(init), domain_width);
}

expr2tc vampire_convt::tuple_get(const expr2tc &expr)
{
  const struct_union_data &strct = get_type_def(expr->type);

  if(is_pointer_type(expr->type))
  {
    // Pointer have two fields, a base address and an offset, so we just
    // need to get the two numbers and call the pointer API

    smt_astt sym = convert_ast(expr);

    smt_astt object = new_ast(
      mk_tuple_select(to_solver_smt_ast<vampire_smt_ast>(sym)->a, 0),
      convert_sort(strct.members[0]));

    smt_astt offset = new_ast(
      mk_tuple_select(to_solver_smt_ast<vampire_smt_ast>(sym)->a, 1),
      convert_sort(strct.members[1]));

    unsigned int num =
      get_bv(object, is_signedbv_type(strct.members[0])).to_uint64();
    unsigned int offs =
      get_bv(offset, is_signedbv_type(strct.members[1])).to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return pointer_logic.back().pointer_expr(p, expr->type);
  }

  // Otherwise, run through all fields and despatch to 'get' again.
  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());
  unsigned int i = 0;
  for(auto const &it : strct.members)
  {
    member2tc memb(it, expr, strct.member_names[i]);
    outstruct->datatype_members.push_back(get(memb));
    i++;
  }

  return outstruct;
}*/

// ***************************** 'get' api *******************************

bool vampire_convt::get_bool(smt_astt a)
{
  /*const vampire_smt_ast *za = to_solver_smt_ast<vampire_smt_ast>(a);
  // Set the model_completion to TRUE.
  // Z3 will assign an interpretation to the Boolean constants,
  // which are essentially don't cares.
  vampire::expr e = solver.get_model().eval(za->a, true);

  Z3_lbool result = Z3_get_bool_value(vampire_ctx, e);

  bool res;
  switch(result)
  {
  case Z3_L_TRUE:
    res = true;
    break;
  case Z3_L_FALSE:
    res = false;
    break;
  default:
    std::cerr << "Can't get boolean value from Z3\n";
    abort();
  }*/

  return false;
}

/*BigInt vampire_convt::get_bv(smt_astt a, bool is_signed)
{
  const vampire_smt_ast *za = to_solver_smt_ast<vampire_smt_ast>(a);
  vampire::expr e = solver.get_model().eval(za->a, true);

  if(int_encoding)
    return string2integer(Z3_get_numeral_string(vampire_ctx, e));

  // Not a numeral? Let's not try to convert it
  return binary2integer(Z3_get_numeral_binary_string(vampire_ctx, e), is_signed);
}

ieee_floatt vampire_convt::get_fpbv(smt_astt a)
{
  const vampire_smt_ast *za = to_solver_smt_ast<vampire_smt_ast>(a);
  vampire::expr e = solver.get_model().eval(za->a, true);

  assert(Z3_get_ast_kind(vampire_ctx, e) == Z3_APP_AST);

  unsigned ew = Z3_fpa_get_ebits(vampire_ctx, e.get_sort());

  // Remove an extra bit added when creating the sort,
  // because we represent the hidden bit like Z3 does
  unsigned sw = Z3_fpa_get_sbits(vampire_ctx, e.get_sort()) - 1;

  ieee_floatt number(ieee_float_spect(sw, ew));
  number.make_zero();

  if(Z3_fpa_is_numeral_nan(vampire_ctx, e))
    number.make_NaN();
  else if(Z3_fpa_is_numeral_inf(vampire_ctx, e))
  {
    if(Z3_fpa_is_numeral_positive(vampire_ctx, e))
      number.make_plus_infinity();
    else
      number.make_minus_infinity();
  }
  else
  {
    Z3_ast v;
    if(Z3_model_eval(
         vampire_ctx, solver.get_model(), Z3_mk_fpa_to_ieee_bv(vampire_ctx, e), 1, &v))
      number.unpack(BigInt(Z3_get_numeral_string(vampire_ctx, v)));
  }

  return number;
}

expr2tc
vampire_convt::get_array_elem(smt_astt array, uint64_t index, const type2tc &subtype)
{
  const vampire_smt_ast *za = to_solver_smt_ast<vampire_smt_ast>(array);
  unsigned long array_bound = array->sort->get_domain_width();
  const vampire_smt_ast *idx;
  if(int_encoding)
    idx = to_solver_smt_ast<vampire_smt_ast>(mk_smt_int(BigInt(index)));
  else
    idx = to_solver_smt_ast<vampire_smt_ast>(
      mk_smt_bv(BigInt(index), mk_bv_sort(array_bound)));

  vampire::expr e = solver.get_model().eval(select(za->a, idx->a), true);
  return get_by_ast(subtype, new_ast(e, convert_sort(subtype)));
}*/

void vampire_smt_ast::dump() const
{
  std::cout << a << std::endl;
  std::cout << std::flush;
}

void vampire_convt::dump_smt()
{
  //std::cout << solver << std::endl;
}

void vampire_convt::print_model()
{
  //std::cout << Z3_model_to_string(vampire_ctx, solver.get_model());
}


smt_sortt vampire_convt::mk_bool_sort()
{
  return new solver_smt_sort<Vampire::Sort>(SMT_SORT_BOOL, solver->boolSort(), 1);
}

smt_sortt vampire_convt::mk_real_sort()
{
  return new solver_smt_sort<Vampire::Sort>(SMT_SORT_REAL, solver->realSort());
}

smt_sortt vampire_convt::mk_int_sort()
{
  return new solver_smt_sort<Vampire::Sort>(SMT_SORT_INT, solver->rationalSort());
}


/*smt_sortt vampire_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<vampire::sort>(domain);
  auto range_sort = to_solver_smt_sort<vampire::sort>(range);

  auto t = vampire_ctx.array_sort(domain_sort->s, range_sort->s);
  return new solver_smt_sort<vampire::sort>(
    SMT_SORT_ARRAY, t, domain->get_data_width(), range);
}

smt_astt vampire_convt::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_to_fp_bv(
        vampire_ctx,
        to_solver_smt_ast<vampire_smt_ast>(op)->a,
        to_solver_smt_sort<vampire::sort>(to)->s)),
    to);
}

smt_astt vampire_convt::mk_from_fp_to_bv(smt_astt op)
{
  smt_sortt to = mk_bvfp_sort(
    op->sort->get_exponent_width(), op->sort->get_significand_width() - 1);
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_to_ieee_bv(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    to);
}

smt_astt
vampire_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const vampire_smt_ast *mrm =
    to_solver_smt_ast<vampire_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);

  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_to_ubv(vampire_ctx, mrm->a, mfrom->a, width)),
    mk_bv_sort(width));
}

smt_astt
vampire_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width)
{
  // Conversion from float to integers always truncate, so we assume
  // the round mode to be toward zero
  const vampire_smt_ast *mrm =
    to_solver_smt_ast<vampire_smt_ast>(mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO));
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);

  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_to_sbv(vampire_ctx, mrm->a, mfrom->a, width)),
    mk_bv_sort(width));
}

smt_astt vampire_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_to_fp_float(
        vampire_ctx, mrm->a, mfrom->a, to_solver_smt_sort<vampire::sort>(to)->s)),
    to);
}

smt_astt
vampire_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_to_fp_unsigned(
        vampire_ctx, mrm->a, mfrom->a, to_solver_smt_sort<vampire::sort>(to)->s)),
    to);
}

smt_astt
vampire_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_to_fp_signed(
        vampire_ctx, mrm->a, mfrom->a, to_solver_smt_sort<vampire::sort>(to)->s)),
    to);
}

smt_astt vampire_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mfrom = to_solver_smt_ast<vampire_smt_ast>(from);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_round_to_integral(vampire_ctx, mrm->a, mfrom->a)),
    from->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mrd = to_solver_smt_ast<vampire_smt_ast>(rd);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_sqrt(vampire_ctx, mrm->a, mrd->a)), rd->sort);
}

smt_astt
vampire_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  const vampire_smt_ast *mv1 = to_solver_smt_ast<vampire_smt_ast>(v1);
  const vampire_smt_ast *mv2 = to_solver_smt_ast<vampire_smt_ast>(v2);
  const vampire_smt_ast *mv3 = to_solver_smt_ast<vampire_smt_ast>(v3);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_fma(vampire_ctx, mrm->a, mv1->a, mv2->a, mv3->a)),
    v1->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const vampire_smt_ast *mlhs = to_solver_smt_ast<vampire_smt_ast>(lhs);
  const vampire_smt_ast *mrhs = to_solver_smt_ast<vampire_smt_ast>(rhs);
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_add(vampire_ctx, mrm->a, mlhs->a, mrhs->a)),
    lhs->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const vampire_smt_ast *mlhs = to_solver_smt_ast<vampire_smt_ast>(lhs);
  const vampire_smt_ast *mrhs = to_solver_smt_ast<vampire_smt_ast>(rhs);
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_sub(vampire_ctx, mrm->a, mlhs->a, mrhs->a)),
    lhs->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const vampire_smt_ast *mlhs = to_solver_smt_ast<vampire_smt_ast>(lhs);
  const vampire_smt_ast *mrhs = to_solver_smt_ast<vampire_smt_ast>(rhs);
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_mul(vampire_ctx, mrm->a, mlhs->a, mrhs->a)),
    lhs->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  const vampire_smt_ast *mlhs = to_solver_smt_ast<vampire_smt_ast>(lhs);
  const vampire_smt_ast *mrhs = to_solver_smt_ast<vampire_smt_ast>(rhs);
  const vampire_smt_ast *mrm = to_solver_smt_ast<vampire_smt_ast>(rm);
  return new_ast(
    vampire::to_expr(vampire_ctx, Z3_mk_fpa_div(vampire_ctx, mrm->a, mlhs->a, mrhs->a)),
    lhs->sort);
}

smt_astt vampire_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_eq(
        vampire_ctx,
        to_solver_smt_ast<vampire_smt_ast>(lhs)->a,
        to_solver_smt_ast<vampire_smt_ast>(rhs)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx, Z3_mk_fpa_is_nan(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_is_infinite(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_is_normal(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx, Z3_mk_fpa_is_zero(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_is_negative(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx,
      Z3_mk_fpa_is_positive(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    boolean_sort);
}

smt_astt vampire_convt::mk_smt_fpbv_abs(smt_astt op)
{
  return new_ast(
    vampire::to_expr(
      vampire_ctx, Z3_mk_fpa_abs(vampire_ctx, to_solver_smt_ast<vampire_smt_ast>(op)->a)),
    op->sort);
}*/
