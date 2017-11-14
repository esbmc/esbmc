#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <sstream>
#include <util/base_type.h>
#include <util/c_types.h>

/* An optimisation of the tuple flattening technique found in smt_tuple_sym.cpp,
 * where we separate out tuple elements into their own variables without any
 * name mangling, and avoid un-necessary operations on elements when the tuple
 * is manipulated.
 *
 * Arrays are handled by using the array flattener API */

void
tuple_node_smt_ast::make_free(smt_convt *ctx)
{
  if (elements.size() != 0)
    return;

  tuple_smt_sortt ts = to_tuple_sort(sort);
  const struct_union_data &strct = ctx->get_type_def(ts->thetype);

  elements.resize(strct.members.size());

  unsigned int i = 0;
  for(auto const &it :strct.members)
  {
    smt_sortt newsort = ctx->convert_sort(it);
    std::string fieldname = name + "." + strct.member_names[i].as_string();

    if (is_tuple_ast_type(it)) {
      elements[i] = ctx->tuple_api->tuple_fresh(newsort, fieldname);
    } else if (is_tuple_array_ast_type(it)) {
      std::string newname = ctx->mk_fresh_name(fieldname);
      smt_sortt subsort = ctx->convert_sort(get_array_subtype(it));
      elements[i] = flat.array_conv.mk_array_symbol(newname, newsort, subsort);
    } else if (is_array_type(it)) {
      elements[i] = ctx->mk_fresh(newsort, fieldname,
                                  ctx->convert_sort(get_array_subtype(it)));
    } else {
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
  tuple_smt_sortt thissort = to_tuple_sort(sort);
  std::string name = ctx->mk_fresh_name("tuple_ite::") + ".";
  tuple_node_smt_ast *result_sym =
    new tuple_node_smt_ast(flat, ctx, sort, name);

  const_cast<tuple_node_smt_ast*>(true_val)->make_free(ctx);
  const_cast<tuple_node_smt_ast*>(false_val)->make_free(ctx);

  const struct_union_data &data = ctx->get_type_def(thissort->thetype);
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

void
tuple_node_smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  // If we're being assigned to something, populate all our vars first
  const_cast<tuple_node_smt_ast*>(this)->make_free(ctx);

  tuple_node_smt_astt target = to_tuple_node_ast(sym);
  assert(target->elements.size() == 0 &&
        "tuple smt assign with elems populated");

  tuple_node_smt_ast *destination = const_cast<tuple_node_smt_ast *>(target);

  // Just copy across element data.
  destination->elements = elements;
}

smt_astt
tuple_node_smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  const_cast<tuple_node_smt_ast*>(to_tuple_node_ast(other))->make_free(ctx);

  // We have two tuple_node_smt_asts and need to create a boolean ast representing
  // their equality: iterate over all their members, compute an equality for
  // each of them, and then combine that into a final ast.
  tuple_node_smt_astt ta = this;
  tuple_node_smt_astt tb = to_tuple_node_ast(other);
  tuple_smt_sortt ts = to_tuple_sort(sort);
  const struct_union_data &data = ctx->get_type_def(ts->thetype);

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
  return ctx->make_conjunct(eqs);
}

smt_astt
tuple_node_smt_ast::update(smt_convt *ctx, smt_astt value, unsigned int idx,
    expr2tc idx_expr __attribute__((unused)) /*ndebug*/) const
{
  smt_convt::ast_vec eqs;
  assert(is_nil_expr(idx_expr) && "Can't apply non-constant index update to "
         "structure");

  std::string name = ctx->mk_fresh_name("tuple_update::") + ".";
  tuple_node_smt_ast *result = new tuple_node_smt_ast(flat, ctx, sort, name);
  result->elements = elements;
  result->make_free(ctx);
  result->elements[idx] = value;

  return result;
}

smt_astt
tuple_node_smt_ast::select(smt_convt *ctx __attribute__((unused)),
    const expr2tc &idx __attribute__((unused))) const
{
  std::cerr << "Select operation applied to tuple" << std::endl;
  abort();
}

smt_astt
tuple_node_smt_ast::project(smt_convt *ctx, unsigned int idx) const
{
  // Create an AST representing the i'th field of the tuple a. This means we
  // have to open up the (tuple symbol) a, tack on the field name to the end
  // of that name, and then return that. It now names the variable that contains
  // the value of that field. If it's actually another tuple, we instead return
  // a new tuple_node_smt_ast containing its name.

  // If someone is projecting out of us, then now is an excellent time to
  // actually allocate all our pieces of ASTs as variables.
  const_cast<tuple_node_smt_ast*>(this)->make_free(ctx);

#ifndef NDEBUG
  tuple_smt_sortt ts = to_tuple_sort(sort);
  const struct_union_data &data = ctx->get_type_def(ts->thetype);
  assert(idx < data.members.size() && "Out-of-bounds tuple element accessed");
#endif
  return elements[idx];
}

smt_astt
smt_tuple_node_flattener::tuple_create(const expr2tc &structdef)
{
  // From a vector of expressions, create a tuple representation by creating
  // a fresh name and assigning members into it.
  std::string name = ctx->mk_fresh_name("tuple_create::");
  // Add a . suffix because this is of tuple type.
  name += ".";

  tuple_node_smt_ast *result =
    new tuple_node_smt_ast(*this, ctx, ctx->convert_sort(structdef->type),
                           name);
  result->elements.resize(structdef->get_num_sub_exprs());

  for (unsigned int i = 0; i < structdef->get_num_sub_exprs(); i++) {
    smt_astt tmp = ctx->convert_ast(*structdef->get_sub_expr(i));
    result->elements[i] = tmp;
  }

  return result;
}

smt_astt
smt_tuple_node_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  if (name == "")
    name = ctx->mk_fresh_name("tuple_fresh::") + ".";

  if (s->id == SMT_SORT_ARRAY) {
    tuple_smt_sortt sort = to_tuple_sort(s);
    assert(is_array_type(sort->thetype));
    smt_sortt subtype = ctx->convert_sort(to_array_type(sort->thetype).subtype);
    return array_conv.mk_array_symbol(name, s, subtype);
  } else {
    return new tuple_node_smt_ast(*this, ctx, s, name);
  }
}

smt_astt
smt_tuple_node_flattener::mk_tuple_symbol(const std::string &name, smt_sortt s)
{

  // Because this tuple flattening doesn't join tuples through the symbol
  // table, there are some special names that need to be intercepted.
  if (name == "0" || name == "NULL")
    return ctx->null_ptr_ast;
  else if (name == "INVALID")
    return ctx->invalid_ptr_ast;

  // We put a '.' on the end of all symbols to deliminate the rest of the
  // name. However, these names may become expressions again, then be converted
  // again, thus accumulating dots. So don't.
  std::string name2 = name;
  if (name2[name2.size() - 1] != '.')
    name2 += ".";

  assert(s->id != SMT_SORT_ARRAY);
  return new tuple_node_smt_ast(*this, ctx, s, name2);
}

smt_astt
smt_tuple_node_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  // Exactly the same as creating a tuple symbol, but for arrays.
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";
  smt_sortt sort = ctx->convert_sort(ctx->flatten_array_type(sym.type));
  smt_sortt subtype = ctx->convert_sort(ctx->get_flattened_array_subtype(sym.type));
  return array_conv.mk_array_symbol(name, sort, subtype);
}

smt_astt
smt_tuple_node_flattener::tuple_array_create(const type2tc &array_type,
                              smt_astt *inputargs,
                              bool const_array,
                              smt_sortt domain)
{
  // Create a tuple array from a constant representation. This means that
  // either we have an array_of or a constant_array. Handle this by creating
  // a fresh tuple array symbol, then repeatedly updating it with tuples at each
  // index. Ignore infinite arrays, they're "not for you".
  // XXX - probably more efficient to update each member array, but not now.
  smt_sortt sort = ctx->convert_sort(array_type);
  smt_sortt subtype = ctx->convert_sort(get_array_subtype(array_type));

  // Optimise the creation of a const array.
  if (const_array)
    return array_conv.convert_array_of_wsort(inputargs[0],
                                             domain->get_data_width(), sort);

  // Otherwise, we'll need to create a new array, and update data into it.
  std::string name = ctx->mk_fresh_name("tuple_array_create::") + ".";
  smt_astt newsym = array_conv.mk_array_symbol(name, sort, subtype);

  // Check size
  const array_type2t &arr_type = to_array_type(array_type);
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return newsym;
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.value.to_ulong();

  // Repeatedly store operands into this.
  for (unsigned int i = 0; i < sz; i++) {
    newsym = newsym->update(ctx, inputargs[i], i);
  }

  return newsym;
}

expr2tc
smt_tuple_node_flattener::tuple_get(const expr2tc &expr)
{
  assert(is_symbol2t(expr) && "Non-symbol in smtlib expr get()");
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name();

  tuple_node_smt_astt a = to_tuple_node_ast(ctx->convert_ast(expr));
  return tuple_get_rec(a);
}

expr2tc
smt_tuple_node_flattener::tuple_get_rec(tuple_node_smt_astt tuple)
{
  tuple_smt_sortt sort = to_tuple_sort(tuple->sort);

  // XXX - what's the correct type to return here.
  constant_struct2tc outstruct(sort->thetype, std::vector<expr2tc>());
  const struct_union_data &strct = ctx->get_type_def(sort->thetype);

  // If this tuple was free and never read, don't attempt to extract data from
  // it. There isn't any.
  if (tuple->elements.size() == 0)
  {
    for(unsigned int i = 0; i < strct.members.size(); i++)
      outstruct->datatype_members.emplace_back();
    return outstruct;
  }

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  for(auto const &it : strct.members)
  {
    expr2tc res;
    if (is_tuple_ast_type(it)) {
      res = tuple_get_rec(to_tuple_node_ast(tuple->elements[i]));
    } else if (is_tuple_array_ast_type(it)) {
      res = expr2tc(); // XXX currently unimplemented
    } else if (is_bool_type(it)) {
      res = ctx->get_bool(tuple->elements[i]);
    } else if (is_number_type(it)) {
      res = ctx->get_bv(it, tuple->elements[i]);
    } else if (is_array_type(it)) {
      std::cerr << "Fetching array elements inside tuples currently unimplemented, sorry" << std::endl;
      res = expr2tc();
    } else {
      std::cerr << "Unexpected type in tuple_get_rec" << std::endl;
      abort();
    }

    outstruct->datatype_members.push_back(res);
    i++;
  }

  // If it's a pointer, rewrite.
  if (is_pointer_type(sort->thetype) || sort->thetype == ctx->pointer_struct) {

    // Guard against a free pointer though
    if (is_nil_expr(outstruct->datatype_members[0]))
      return expr2tc();

    uint64_t num = to_constant_int2t(outstruct->datatype_members[0])
                                    .value.to_uint64();
    uint64_t offs = to_constant_int2t(outstruct->datatype_members[1])
                                     .value.to_uint64();
    pointer_logict::pointert p(num, BigInt(offs));
    return ctx->pointer_logic.back().pointer_expr(p,
                                 type2tc(new pointer_type2t(get_empty_type())));
  }

  return outstruct;
}

smt_astt
smt_tuple_node_flattener::tuple_array_of(const expr2tc &init_val, unsigned long array_size)
{

  uint64_t elems = 1ULL << array_size;
  type2tc array_type =
    type2tc(new array_type2t(init_val->type, gen_ulong(elems), false));
  smt_sortt array_sort = new tuple_smt_sort(array_type, 1, array_size);

  return array_conv.convert_array_of_wsort(ctx->convert_ast(init_val),
    array_size, array_sort);
}

smt_sortt
smt_tuple_node_flattener::mk_struct_sort(const type2tc &type)
{

  if (is_array_type(type)) {
    const array_type2t &arrtype = to_array_type(type);
    assert(!is_array_type(arrtype.subtype) && "Arrays dimensions should be flattened by the time they reach tuple interface");
    unsigned int dom_width = ctx->calculate_array_domain_width(arrtype);
    // NB: the range value is a dummy.
    return new tuple_smt_sort(type, 1, dom_width);
  } else {
    return new tuple_smt_sort(type);
  }
}

void
smt_tuple_node_flattener::add_tuple_constraints_for_solving()
{
  array_conv.add_array_constraints_for_solving();
}

void
smt_tuple_node_flattener::push_tuple_ctx()
{
  array_conv.push_array_ctx();
}

void
smt_tuple_node_flattener::pop_tuple_ctx()
{
  array_conv.pop_array_ctx();
}
