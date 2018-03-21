#include <solvers/smt/smt_conv.h>
#include <solvers/smt/tuple/smt_tuple_array_ast.h>
#include <solvers/smt/tuple/smt_tuple_sym_ast.h>
#include <solvers/smt/tuple/smt_tuple_sym.h>
#include <sstream>
#include <util/base_type.h>
#include <util/c_types.h>

smt_astt smt_tuple_sym_flattener::tuple_create(const expr2tc &structdef)
{
  // From a vector of expressions, create a tuple representation by creating
  // a fresh name and assigning members into it.
  std::string name = ctx->mk_fresh_name("tuple_create::");

  // Add a . suffix because this is of tuple type.
  name += ".";

  smt_ast *result =
    new tuple_sym_smt_ast(ctx, ctx->convert_sort(structdef->type), name);

  for(unsigned int i = 0; i < structdef->get_num_sub_exprs(); i++)
  {
    smt_astt tmp = ctx->convert_ast(*structdef->get_sub_expr(i));
    smt_astt elem = result->project(ctx, i);
    ctx->assert_ast(elem->eq(ctx, tmp));
  }

  return result;
}

smt_astt smt_tuple_sym_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  std::string n =
    (name == "") ? ctx->mk_fresh_name("tuple_fresh::") + "." : name;

  if(s->id == SMT_SORT_ARRAY)
    return new array_sym_smt_ast(ctx, s, n);

  return new tuple_sym_smt_ast(ctx, s, n);
}

smt_astt
smt_tuple_sym_flattener::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  // We put a '.' on the end of all symbols to deliminate the rest of the
  // name. However, these names may become expressions again, then be converted
  // again, thus accumulating dots. So don't.
  std::string name2 = name;
  if(name2[name2.size() - 1] != '.')
    name2 += ".";

  assert(s->id != SMT_SORT_ARRAY);
  return new tuple_sym_smt_ast(ctx, s, name2);
}

smt_astt smt_tuple_sym_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  // Exactly the same as creating a tuple symbol, but for arrays.
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";
  smt_sortt sort = ctx->convert_sort(sym.type);
  return new array_sym_smt_ast(ctx, sort, name);
}

smt_astt smt_tuple_sym_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain __attribute__((unused)))
{
  // Create a tuple array from a constant representation. This means that
  // either we have an array_of or a constant_array. Handle this by creating
  // a fresh tuple array symbol, then repeatedly updating it with tuples at each
  // index. Ignore infinite arrays, they're "not for you".
  // XXX - probably more efficient to update each member array, but not now.
  smt_sortt sort = ctx->convert_sort(array_type);
  std::string name = ctx->mk_fresh_name("tuple_array_create::") + ".";
  smt_astt newsym = new array_sym_smt_ast(ctx, sort, name);

  // Check size
  const array_type2t &arr_type = to_array_type(array_type);
  if(arr_type.size_is_infinite)
  {
    // Guarentee nothing, this is modelling only.
    return newsym;
  }
  if(!is_constant_int2t(arr_type.array_size))
  {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.value.to_ulong();

  if(const_array)
  {
    // Repeatedly store the same value into this at all the demanded
    // indexes.
    smt_astt init = inputargs[0];
    for(unsigned int i = 0; i < sz; i++)
    {
      newsym = newsym->update(ctx, init, i);
    }

    return newsym;
  }

  // Repeatedly store operands into this.
  for(unsigned int i = 0; i < sz; i++)
  {
    newsym = newsym->update(ctx, inputargs[i], i);
  }

  return newsym;
}

expr2tc smt_tuple_sym_flattener::tuple_get(const expr2tc &expr)
{
  assert(is_symbol2t(expr) && "Non-symbol in smtlib expr get()");
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name();

  const type2tc &thetype =
    (is_structure_type(expr->type)) ? expr->type : ctx->pointer_struct;
  const struct_union_data &strct = ctx->get_type_def(thetype);

  // XXX - what's the correct type to return here.
  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  for(auto const &it : strct.members)
  {
    std::stringstream ss;
    ss << name << "." << strct.member_names[i];
    symbol2tc sym(it, ss.str());
    outstruct->datatype_members.push_back(ctx->get(sym));
    i++;
  }

  // If it's a pointer, rewrite.
  if(is_pointer_type(expr->type))
  {
    // Guard against free pointer value
    if(is_nil_expr(outstruct->datatype_members[0]))
      return expr2tc();

    uint64_t num =
      to_constant_int2t(outstruct->datatype_members[0]).value.to_uint64();
    uint64_t offs =
      to_constant_int2t(outstruct->datatype_members[1]).value.to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return ctx->pointer_logic.back().pointer_expr(p, expr->type);
  }

  return outstruct;
}

smt_astt smt_tuple_sym_flattener::tuple_array_of(
  const expr2tc &init_val,
  unsigned long array_size)
{
  // An array of tuples without tuple support: decompose into array_of's each
  // subtype.
  const struct_union_data &subtype = ctx->get_type_def(init_val->type);
  const constant_datatype_data &data =
    static_cast<const constant_datatype_data &>(*init_val.get());

  constant_int2tc arrsize(index_type2(), BigInt(array_size));
  type2tc arrtype(new array_type2t(init_val->type, arrsize, false));
  std::string name = ctx->mk_fresh_name("tuple_array_of::") + ".";
  symbol2tc tuple_arr_of_sym(arrtype, irep_idt(name));

  smt_sortt sort = ctx->convert_sort(arrtype);
  smt_astt newsym = new array_sym_smt_ast(ctx, sort, name);

  assert(subtype.members.size() == data.datatype_members.size());
  for(unsigned long i = 0; i < subtype.members.size(); i++)
  {
    const expr2tc &val = data.datatype_members[i];
    type2tc subarr_type = array_type2tc(val->type, arrsize, false);
    constant_array_of2tc sub_array_of(subarr_type, val);

    smt_astt tuple_arr_of_sym_ast = ctx->convert_ast(tuple_arr_of_sym);
    smt_astt target_array = tuple_arr_of_sym_ast->project(ctx, i);

    smt_astt sub_array_of_ast = ctx->convert_ast(sub_array_of);
    ctx->assert_ast(target_array->eq(ctx, sub_array_of_ast));
  }

  return newsym;
}

smt_sortt smt_tuple_sym_flattener::mk_struct_sort(const type2tc &type)
{
  if(is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    assert(
      !is_array_type(arrtype.subtype) &&
      "Arrays dimensions should be flattened by the time they reach tuple "
      "interface");
    unsigned int dom_width = ctx->calculate_array_domain_width(arrtype);
    return new smt_sort(
      SMT_SORT_ARRAY, type, dom_width, ctx->convert_sort(arrtype.subtype));
  }

  return new smt_sort(SMT_SORT_STRUCT, type);
}

void smt_tuple_sym_flattener::add_tuple_constraints_for_solving()
{
}

void smt_tuple_sym_flattener::push_tuple_ctx()
{
}

void smt_tuple_sym_flattener::pop_tuple_ctx()
{
}
