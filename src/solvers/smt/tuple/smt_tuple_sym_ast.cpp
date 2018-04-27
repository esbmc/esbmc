#include <solvers/smt/smt_conv.h>
#include <solvers/smt/tuple/smt_tuple_array_ast.h>
#include <solvers/smt/tuple/smt_tuple_sym.h>
#include <solvers/smt/tuple/smt_tuple_sym_ast.h>
#include <sstream>
#include <util/base_type.h>
#include <util/c_types.h>

/** @file smt_tuple.cpp
 * So, the SMT-encoding-with-no-tuple-support. SMT itself doesn't support
 * tuples, but we've been pretending that it does by using Z3 almost
 * exclusively (which does support tuples). So, we have to find some way of
 * supporting:
 *
 *  1) Tuples
 *  2) Arrays of tuples
 *  3) Arrays of tuples containing arrays
 *
 * 1: In all circumstances we're either creating, projecting, or updating,
 *    a set of variables that are logically grouped together as a tuple. We
 *    can generally just handle that by linking up a tuple to the actual
 *    variable that we want. To do this, we create a set of symbols /underneath/
 *    the symbol we're dealing with, corresponding to the field name. So a tuple
 *    with fields a, b, and c, with the symbol name "faces" would create:
 *
 *      main::1::faces.a
 *      main::1::faces.b
 *      main::1::faces.c
 *
 *    As variables with the appropriate type. Project / update redirects
 *    expressions to deal with those symbols. Equality is similar.
 *
 *    It gets more complicated with ite's though, as you can't
 *    nondeterministically switch between symbol prefixes like that. So instead
 *    we create a fresh new symbol, and create an ite for each member of the
 *    tuple, binding it into the new fresh symbol if it's enabling condition
 *    is true.
 *
 *    The basic design feature here is that in all cases where we have something
 *    of tuple type, the AST is actually a deterministic symbol that we hang
 *    additional bits of name off as appropriate.
 *
 *  2: For arrays of tuples, we do almost exactly as above, but all the new
 *     variables that are used are in fact arrays of values. Then when we
 *     perform an array operation, we either select the relevant values out into
 *     a fresh tuple, or decompose a tuple into a series of updates to make.
 *
 *  3: Tuples of arrays of tuples are currently unimplemented. But it's likely
 *     that we'll end up following 2: above, but extending the domain of the
 *     array to contain the outer and inner array index. Ugly but works
 *     (inspiration came from the internet / stackoverflow, reading other
 *     peoples work where they've done this).
 *
 * All these things could probably be made more efficient by rewriting the
 * expressions that are being converted, so that we can for example discard
 * any un-necessary equalities or assertions. However, in the meantime, this
 * slower approach works.
 */

smt_astt
tuple_sym_smt_ast::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  // So - we need to generate an ite between true_val and false_val, that gets
  // switched on based on cond, and store the output into result. Do this by
  // projecting each member out of our arguments and computing another ite
  // over each member. Note that we always make assertions here, because the
  // ite is always true. We return the output symbol.
  tuple_sym_smt_astt true_val = this;
  tuple_sym_smt_astt false_val = to_tuple_sym_ast(falseop);

  std::string name = ctx->mk_fresh_name("tuple_ite::") + ".";
  symbol2tc result(sort->get_tuple_type(), name);
  smt_astt result_sym = ctx->convert_ast(result);

  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());

  // Iterate through each field and encode an ite.
  for(unsigned int i = 0; i < data.members.size(); i++)
  {
    smt_astt truepart = true_val->project(ctx, i);
    smt_astt falsepart = false_val->project(ctx, i);

    smt_astt result_ast = truepart->ite(ctx, cond, falsepart);

    smt_astt result_sym_ast = result_sym->project(ctx, i);
    ctx->assert_ast(result_sym_ast->eq(ctx, result_ast));
  }

  return ctx->convert_ast(result);
}

smt_astt tuple_sym_smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  // We have two tuple_sym_smt_asts and need to create a boolean ast representing
  // their equality: iterate over all their members, compute an equality for
  // each of them, and then combine that into a final ast.
  tuple_sym_smt_astt ta = this;
  tuple_sym_smt_astt tb = to_tuple_sym_ast(other);

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

smt_astt tuple_sym_smt_ast::update(
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

  // XXX: future work, accept member_name exprs?
  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());

  std::string name = ctx->mk_fresh_name("tuple_update::") + ".";
  tuple_sym_smt_astt result = new tuple_sym_smt_ast(ctx, sort, name);

  // Iterate over all members, deciding what to do with them.
  for(unsigned int j = 0; j < data.members.size(); j++)
  {
    if(j == idx)
    {
      // This is the updated field -- generate the name of its variable with
      // tuple project and assign it in.
      smt_astt thefield = result->project(ctx, j);

      eqs.push_back(thefield->eq(ctx, value));
    }
    else
    {
      // This is not an updated field; extract the member out of the input
      // tuple (a) and assign it into the fresh tuple.
      smt_astt field1 = project(ctx, j);
      smt_astt field2 = result->project(ctx, j);
      eqs.push_back(field1->eq(ctx, field2));
    }
  }

  ctx->assert_ast(ctx->make_n_ary(ctx, &smt_convt::mk_and, eqs));
  return result;
}

smt_astt tuple_sym_smt_ast::select(
  smt_convt *ctx __attribute__((unused)),
  const expr2tc &idx __attribute__((unused))) const
{
  std::cerr << "Select operation applied to tuple" << std::endl;
  abort();
}

smt_astt tuple_sym_smt_ast::project(smt_convt *ctx, unsigned int idx) const
{
  // Create an AST representing the i'th field of the tuple a. This means we
  // have to open up the (tuple symbol) a, tack on the field name to the end
  // of that name, and then return that. It now names the variable that contains
  // the value of that field. If it's actually another tuple, we instead return
  // a new tuple_sym_smt_ast containing its name.
  const struct_union_data &data = ctx->get_type_def(sort->get_tuple_type());

  assert(idx < data.members.size() && "Out-of-bounds tuple element accessed");
  const std::string &fieldname = data.member_names[idx].as_string();
  std::string sym_name = name + fieldname;

  // Cope with recursive structs.
  const type2tc &restype = data.members[idx];
  smt_sortt s = ctx->convert_sort(restype);

  if(is_tuple_ast_type(restype) || is_tuple_array_ast_type(restype))
  {
    // This is a struct within a struct, so just generate the name prefix of
    // the internal struct being projected.
    sym_name = sym_name + ".";
    if(is_tuple_array_ast_type(restype))
      return new array_sym_smt_ast(ctx, s, sym_name);

    return new tuple_sym_smt_ast(ctx, s, sym_name);
  }
  else
  {
    // This is a normal variable, so create a normal symbol of its name.
    return ctx->mk_smt_symbol(sym_name, s);
  }
}
