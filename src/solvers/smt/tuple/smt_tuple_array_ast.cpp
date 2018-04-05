#include <solvers/smt/smt_conv.h>
#include <solvers/smt/tuple/smt_tuple_array_ast.h>
#include <sstream>
#include <util/base_type.h>
#include <util/c_types.h>

smt_astt
array_sym_smt_ast::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  // Similar to tuple ite's, but the leafs are arrays.
  tuple_sym_smt_astt true_val = this;
  tuple_sym_smt_astt false_val = to_tuple_sym_ast(falseop);

  assert(is_array_type(sort->get_tuple_type()));
  const array_type2t &array_type = to_array_type(sort->get_tuple_type());

  std::string name = ctx->mk_fresh_name("tuple_array_ite::") + ".";
  symbol2tc result(sort->get_tuple_type(), name);
  smt_astt result_sym = ctx->convert_ast(result);

  const struct_union_data &data = ctx->get_type_def(array_type.subtype);

  // Iterate through each field and encode an ite.
  unsigned int i = 0;
  for(auto const &it : data.members)
  {
    array_type2tc arrtype(
      it, array_type.array_size, array_type.size_is_infinite);

    smt_astt truepart = true_val->project(ctx, i);
    smt_astt falsepart = false_val->project(ctx, i);

    smt_astt result_ast = truepart->ite(ctx, cond, falsepart);

    smt_astt result_sym_ast = result_sym->project(ctx, i);

    ctx->assert_ast(result_sym_ast->eq(ctx, result_ast));
    i++;
  }

  return ctx->convert_ast(result);
}

smt_astt array_sym_smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  // We have two tuple_sym_smt_asts and need to create a boolean ast representing
  // their equality: iterate over all their members, compute an equality for
  // each of them, and then combine that into a final ast.
  tuple_sym_smt_astt ta = this;
  tuple_sym_smt_astt tb = to_tuple_sym_ast(other);
  assert(is_array_type(sort->get_tuple_type()));
  const array_type2t &arrtype = to_array_type(sort->get_tuple_type());
  const struct_union_data &data = ctx->get_type_def(arrtype.subtype);

  smt_convt::ast_vec eqs;
  eqs.reserve(data.members.size());

  // Iterate through each field and encode an equality.
  unsigned int i = 0;
  for(auto const &it : data.members)
  {
    type2tc tmparrtype(
      new array_type2t(it, arrtype.array_size, arrtype.size_is_infinite));
    smt_astt side1 = ta->project(ctx, i);
    smt_astt side2 = tb->project(ctx, i);
    eqs.push_back(side1->eq(ctx, side2));
    i++;
  }

  // Create an ast representing the fact that all the members are equal.
  return ctx->make_n_ary(ctx, &smt_convt::mk_and, eqs);
}

smt_astt array_sym_smt_ast::update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  const array_type2t array_type = to_array_type(sort->get_tuple_type());
  const struct_union_data &data = ctx->get_type_def(array_type.subtype);

  expr2tc index;
  if(is_nil_expr(idx_expr))
  {
    index =
      constant_int2tc(ctx->make_array_domain_type(array_type), BigInt(idx));
  }
  else
  {
    index = idx_expr;
  }

  std::string name = ctx->mk_fresh_name("tuple_array_update::") + ".";
  tuple_sym_smt_astt result = new array_sym_smt_ast(ctx, sort, name);

  // Iterate over all members. They are _all_ indexed and updated.
  unsigned int i = 0;
  for(auto const &it : data.members)
  {
    type2tc arrtype(
      new array_type2t(it, array_type.array_size, array_type.size_is_infinite));

    // Project and update a field in 'this'
    smt_astt field = project(ctx, i);
    smt_astt resval = value->project(ctx, i);
    smt_astt updated = field->update(ctx, resval, 0, index);

    // Now equality it into the result object
    smt_astt res_field = result->project(ctx, i);
    updated->assign(ctx, res_field);

    i++;
  }

  return result;
}

smt_astt array_sym_smt_ast::select(smt_convt *ctx, const expr2tc &idx) const
{
  const array_type2t &array_type = to_array_type(sort->get_tuple_type());
  const struct_union_data &data = ctx->get_type_def(array_type.subtype);
  smt_sortt result_sort = ctx->convert_sort(array_type.subtype);

  std::string name = ctx->mk_fresh_name("tuple_array_select::") + ".";
  tuple_sym_smt_astt result = new tuple_sym_smt_ast(ctx, result_sort, name);

  unsigned int i = 0;
  for(auto const &it : data.members)
  {
    type2tc arrtype(
      new array_type2t(it, array_type.array_size, array_type.size_is_infinite));

    smt_astt result_field = result->project(ctx, i);
    smt_astt sub_array = project(ctx, i);

    smt_astt selected = sub_array->select(ctx, idx);
    ctx->assert_ast(result_field->eq(ctx, selected));

    i++;
  }

  return result;
}

smt_astt array_sym_smt_ast::project(smt_convt *ctx, unsigned int idx) const
{
  // Pull struct type out, access the relevent element, then wrap it in an
  // array type.

  const array_type2t &arr = to_array_type(sort->get_tuple_type());
  const struct_union_data &data = ctx->get_type_def(arr.subtype);

  assert(
    idx < data.members.size() && "Out-of-bounds tuple-array element accessed");
  const std::string &fieldname = data.member_names[idx].as_string();
  std::string sym_name = name + fieldname;

  const type2tc &restype = data.members[idx];
  type2tc new_arr_type(
    new array_type2t(restype, arr.array_size, arr.size_is_infinite));
  smt_sortt s = ctx->convert_sort(new_arr_type);

  if(is_tuple_ast_type(restype) || is_tuple_array_ast_type(restype))
  {
    // This is a struct within a struct, so just generate the name prefix of
    // the internal struct being projected.
    sym_name = sym_name + ".";
    return new array_sym_smt_ast(ctx, s, sym_name);
  }

  // This is a normal variable, so create a normal symbol of its name.
  return ctx->mk_smt_symbol(sym_name, s);
}

void array_sym_smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  // We have two tuple_sym_smt_asts and need to call assign on all of their
  // components.
  array_sym_smt_astt src = this;
  array_sym_smt_astt dst = to_array_sym_ast(sym);

  const array_type2t &arrtype = to_array_type(sort->get_tuple_type());
  const struct_union_data &data = ctx->get_type_def(arrtype.subtype);

  unsigned int i = 0;
  for(auto const &it : data.members)
  {
    array_type2tc tmparrtype(it, arrtype.array_size, arrtype.size_is_infinite);
    smt_astt source = src->project(ctx, i);
    smt_astt destination = dst->project(ctx, i);
    source->assign(ctx, destination);
    i++;
  }
}
