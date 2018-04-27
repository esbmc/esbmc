#include <solvers/smt/smt_conv.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_node.h>
#include <solvers/smt/tuple/smt_tuple_node_ast.h>
#include <sstream>
#include <util/base_type.h>
#include <util/c_types.h>

/* An optimisation of the tuple flattening technique found in smt_tuple_sym.cpp,
 * where we separate out tuple elements into their own variables without any
 * name mangling, and avoid un-necessary operations on elements when the tuple
 * is manipulated.
 *
 * Arrays are handled by using the array flattener API */

void tuple_node_smt_ast::make_free(smt_convt *ctx)
{
  if(elements.size() != 0)
    return;

  const struct_union_data &strct = ctx->get_type_def(sort->get_tuple_type());

  elements.resize(strct.members.size());

  unsigned int i = 0;
  for(auto const &it : strct.members)
  {
    smt_sortt newsort = ctx->convert_sort(it);
    std::string fieldname = name + "." + strct.member_names[i].as_string();

    if(is_tuple_ast_type(it))
    {
      elements[i] = ctx->tuple_api->tuple_fresh(newsort, fieldname);
    }
    else if(is_tuple_array_ast_type(it))
    {
      std::string newname = ctx->mk_fresh_name(fieldname);
      smt_sortt subsort = ctx->convert_sort(get_array_subtype(it));
      elements[i] = flat.array_conv.mk_array_symbol(newname, newsort, subsort);
    }
    else if(is_array_type(it))
    {
      elements[i] = ctx->mk_fresh(
        newsort, fieldname, ctx->convert_sort(get_array_subtype(it)));
    }
    else
    {
      elements[i] = ctx->mk_fresh(newsort, fieldname);
    }

    i++;
  }
}

smt_astt
tuple_node_smt_ast::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  // So - we need to generate an ite between true_val and false_val, that gets
  // switched on based on cond, and store the output into result. Do this by
  // projecting each member out of our arguments and computing another ite
  // over each member. Note that we always make assertions here, because the
  // ite is always true. We return the output symbol.
  tuple_node_smt_astt true_val = this;
  tuple_node_smt_astt false_val = to_tuple_node_ast(falseop);

  std::string name = ctx->mk_fresh_name("tuple_ite::") + ".";
  tuple_node_smt_ast *result_sym =
    new tuple_node_smt_ast(flat, ctx, sort, name);

  const_cast<tuple_node_smt_ast *>(true_val)->make_free(ctx);
  const_cast<tuple_node_smt_ast *>(false_val)->make_free(ctx);

  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());
  result_sym->elements.resize(data.members.size());

  // Iterate through each field and encode an ite.
  for(unsigned int i = 0; i < data.members.size(); i++)
  {
    smt_astt truepart = true_val->project(ctx, i);
    smt_astt falsepart = false_val->project(ctx, i);

    smt_astt result_ast = truepart->ite(ctx, cond, falsepart);

    result_sym->elements[i] = result_ast;
  }

  return result_sym;
}

void tuple_node_smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  // If we're being assigned to something, populate all our vars first
  const_cast<tuple_node_smt_ast *>(this)->make_free(ctx);

  tuple_node_smt_astt target = to_tuple_node_ast(sym);
  assert(
    target->elements.size() == 0 && "tuple smt assign with elems populated");

  tuple_node_smt_ast *destination = const_cast<tuple_node_smt_ast *>(target);

  // Just copy across element data.
  destination->elements = elements;
}

smt_astt tuple_node_smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  const_cast<tuple_node_smt_ast *>(to_tuple_node_ast(other))->make_free(ctx);

  // We have two tuple_node_smt_asts and need to create a boolean ast representing
  // their equality: iterate over all their members, compute an equality for
  // each of them, and then combine that into a final ast.
  tuple_node_smt_astt ta = this;
  tuple_node_smt_astt tb = to_tuple_node_ast(other);

  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());

  smt_convt::ast_vec eqs;
  eqs.reserve(data.members.size());

  // Iterate through each field and encode an equality.
  for(unsigned int i = 0; i < data.members.size(); i++)
  {
    smt_astt side1 = ta->project(ctx, i);
    smt_astt side2 = tb->project(ctx, i);
    eqs.push_back(side1->eq(ctx, side2));
  }

  // Create an ast representing the fact that all the members are equal.
  return ctx->make_n_ary(ctx, &smt_convt::mk_and, eqs);
}

smt_astt tuple_node_smt_ast::update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr __attribute__((unused)) /*ndebug*/) const
{
  smt_convt::ast_vec eqs;
  assert(
    is_nil_expr(idx_expr) &&
    "Can't apply non-constant index update to "
    "structure");

  std::string name = ctx->mk_fresh_name("tuple_update::") + ".";
  tuple_node_smt_ast *result = new tuple_node_smt_ast(flat, ctx, sort, name);
  result->elements = elements;
  result->make_free(ctx);
  result->elements[idx] = value;

  return result;
}

smt_astt tuple_node_smt_ast::select(
  smt_convt *ctx __attribute__((unused)),
  const expr2tc &idx __attribute__((unused))) const
{
  std::cerr << "Select operation applied to tuple" << std::endl;
  abort();
}

smt_astt tuple_node_smt_ast::project(smt_convt *ctx, unsigned int idx) const
{
  // Create an AST representing the i'th field of the tuple a. This means we
  // have to open up the (tuple symbol) a, tack on the field name to the end
  // of that name, and then return that. It now names the variable that contains
  // the value of that field. If it's actually another tuple, we instead return
  // a new tuple_node_smt_ast containing its name.

  // If someone is projecting out of us, then now is an excellent time to
  // actually allocate all our pieces of ASTs as variables.
  const_cast<tuple_node_smt_ast *>(this)->make_free(ctx);

#ifndef NDEBUG
  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());
  assert(idx < data.members.size() && "Out-of-bounds tuple element accessed");
#endif
  return elements[idx];
}
