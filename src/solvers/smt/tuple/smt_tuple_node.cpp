#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_node_ast.h>
#include <solvers/smt/tuple/smt_tuple_node.h>
#include <sstream>
#include <util/expr/base_type.h>
#include <util/lang/c_types.h>

smt_astt smt_tuple_node_flattener::tuple_create(const expr2tc &structdef)
{
  // From a vector of expressions, create a tuple representation by creating
  // a fresh name and assigning members into it.
  std::string name = ctx->mk_fresh_name("tuple_create::");
  // Add a . suffix because this is of tuple type.
  name += ".";

  tuple_node_smt_ast *result = new tuple_node_smt_ast(
    *this, ctx, ctx->convert_sort(structdef->type), name);
  result->elements.resize(structdef->get_num_sub_exprs());

  for (size_t i = 0; i < structdef->get_num_sub_exprs(); i++)
  {
    smt_astt tmp = ctx->convert_ast(*structdef->get_sub_expr(i));
    result->elements[i] = tmp;
  }

  return result;
}

smt_astt smt_tuple_node_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  if (name == "")
    name = ctx->mk_fresh_name("tuple_fresh::") + ".";

  if (s->id == SMT_SORT_ARRAY)
  {
    assert(is_array_type(s->get_tuple_type()));
    smt_sortt subtype =
      ctx->convert_sort(to_array_type(s->get_tuple_type()).subtype);
    return array_conv.mk_array_symbol(name, s, subtype);
  }

  return new tuple_node_smt_ast(*this, ctx, s, name);
}

smt_astt
smt_tuple_node_flattener::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  // Because this tuple flattening doesn't join tuples through the symbol
  // table, there are some special names that need to be intercepted.
  if (name == "NULL")
    return ctx->null_ptr_ast;

  if (name == "INVALID")
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

// 2C.2c: rebuild a K-dim array_type chain with its leaf replaced by
// `leaf` (each dimension's size / size_is_infinite preserved).  Used to
// derive the per-field native array type array^K<fi> from array^K<S>.
static type2tc rebuild_array_leaf(const type2tc &t, const type2tc &leaf_type)
{
  if (!is_array_type(t))
    return leaf_type;
  const array_type2t &a = to_array_type(t);
  return array_type2tc(
    rebuild_array_leaf(a.subtype, leaf_type),
    a.array_size,
    a.size_is_infinite,
    a.index_width);
}

smt_astt smt_tuple_node_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";

  // Walk the array dimensions to the leaf.  K = 1 (the immediate
  // subtype is the struct itself, not an array) keeps the historical
  // single-level array_conv route verbatim → byte-identical.
  // Only a chain with an infinite level (Solidity's nested mappings) takes
  // the struct-of-arrays route: a finite one, a VLA included, flattens to a
  // single array like every other finite array of arrays.
  const array_type2t &outer = to_array_type(sym.type);
  bool k_ge_2 =
    is_array_type(outer.subtype) && array_chain_has_infinite_level(sym.type);

  type2tc leaf = sym.type;
  while (is_array_type(leaf))
    leaf = to_array_type(leaf).subtype;

  if (!k_ge_2 || !is_struct_type(leaf))
  {
    // K=1 array-of-struct, or a non-struct leaf (pointer/code/complex
    // nested array — out of scope for the SoA decomposition): unchanged.
    smt_sortt sort = ctx->convert_sort(ctx->flatten_array_type(sym.type));
    smt_sortt subtype =
      ctx->convert_sort(ctx->get_flattened_array_subtype(sym.type));
    return array_conv.mk_array_symbol(name, sort, subtype);
  }

  // 2C.2c — K>=2 struct leaf: struct-of-arrays representation.  A
  // tuple_node whose sort is the leaf struct (project / eq / get walk
  // its m members) and whose elements[i] is a solver-NATIVE array
  // array^K<fi> (primitive fi ⇒ Branch A native, no bare sort, no
  // array_conv; struct fi ⇒ convert_ast re-enters this builder and
  // recurses).  No read/write semantics change here — that is 2C.2d.
  smt_sortt tsort = ctx->convert_sort(leaf);
  tuple_node_smt_ast *result = new tuple_node_smt_ast(*this, ctx, tsort, name);

  const std::vector<type2tc> &leaf_members = struct_union_members(leaf);
  result->elements.resize(leaf_members.size());

  unsigned int i = 0;
  for (auto const &it : leaf_members)
  {
    type2tc fld_arr_type = rebuild_array_leaf(sym.type, it);
    std::string fname =
      name + "." + struct_union_member_names(leaf)[i].as_string();
    result->elements[i] =
      ctx->convert_ast(symbol2tc(fld_arr_type, irep_idt(fname)));
    i++;
  }

  return result;
}

/** Reads an unbounded constant array of structs whose members are all
 *  bit-vectors through one native solver array per member, each a chain of
 *  stores over that member's values, so a read is one select per member.
 *  Returns false, and does nothing, when the solver has no native arrays or a
 *  member is not a bit-vector. */
static bool read_through_member_arrays(
  smt_tuple_node_flattener &flat,
  smt_solver_baset *ctx,
  array_convt &array_conv,
  smt_astt array,
  smt_astt *vals,
  unsigned int sz,
  smt_sortt subtype)
{
  if (
    ctx->array_api == nullptr ||
    dynamic_cast<array_convt *>(ctx->array_api) != nullptr ||
    ctx->int_encoding || sz == 0)
    return false;

  size_t n = struct_union_members(subtype->get_tuple_type()).size();
  for (size_t m = 0; m < n; m++)
  {
    smt_sortt ms = vals[0]->project(ctx, m)->sort;
    if (ms->id != SMT_SORT_BV && ms->id != SMT_SORT_FIXEDBV)
      return false;
  }

  size_t dom_width = array->sort->get_domain_width();
  smt_sortt dom_sort = ctx->mk_int_bv_sort(dom_width);
  std::vector<smt_astt> arrays;
  for (size_t m = 0; m < n; m++)
  {
    smt_sortt ms = vals[0]->project(ctx, m)->sort;
    smt_astt a = ctx->array_api->mk_array_symbol(
      ctx->mk_fresh_name("tuple_array_member::"),
      ctx->mk_array_sort(dom_sort, ms),
      ms);
    for (unsigned int i = 0; i < sz; i++)
      a = ctx->mk_store(
        a, ctx->mk_smt_bv(BigInt(i), dom_width), vals[i]->project(ctx, m));
    arrays.push_back(a);
  }

  array_conv.set_initial_reader(
    array, [&flat, ctx, subtype, arrays](const expr2tc &idx) -> smt_astt {
      smt_astt real_idx = ctx->convert_ast(idx);
      std::string name = ctx->mk_fresh_name("tuple_array_member_read::") + ".";
      auto *res = new tuple_node_smt_ast(flat, ctx, subtype, name);
      for (smt_astt a : arrays)
        res->elements.push_back(ctx->mk_select(a, real_idx));
      return res;
    });
  return true;
}

smt_astt smt_tuple_node_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain)
{
  // Create a tuple array from a constant representation. This means that
  // either we have an array_of or a constant_array. Handle this by creating
  // a fresh tuple array symbol, then repeatedly updating it with tuples at each
  // index. Ignore infinite arrays, they're "not for you".
  smt_sortt sort = ctx->convert_sort(array_type);
  smt_sortt subtype = ctx->convert_sort(to_array_type(array_type).subtype);

  // Optimize the creation of a const array.
  if (const_array)
    return array_conv.convert_array_of_wsort(
      inputargs[0], domain->get_data_width(), sort);

  // Otherwise, we'll need to create a new array, and update data into it.
  std::string name = ctx->mk_fresh_name("tuple_array_create::") + ".";
  smt_astt newsym = array_conv.mk_array_symbol(name, sort, subtype);

  // Check size
  const array_type2t &arr_type = to_array_type(array_type);
  if (arr_type.size_is_infinite)
  {
    // Guarentee nothing, this is modelling only.
    return newsym;
  }
  if (!is_constant_int2t(arr_type.array_size))
  {
    log_error("Non-constant sized array of type constant_array_of2t");
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  unsigned int sz = thesize.value.to_uint64();

  // Above the bounded-array cutoff, the array flattener re-executes each
  // update at every index the array is read at, which is quadratic in the
  // number of elements. Read through native member arrays instead if we can.
  if (
    is_unbounded_array(sort) &&
    read_through_member_arrays(
      *this, ctx, array_conv, newsym, inputargs, sz, subtype))
    return newsym;

  // Repeatedly store operands into this.
  for (unsigned int i = 0; i < sz; i++)
  {
    newsym = newsym->update(ctx, inputargs[i], i);
  }

  return newsym;
}

expr2tc smt_tuple_node_flattener::tuple_get(const type2tc &, smt_astt sym)
{
  return tuple_get_rec(to_tuple_node_ast(sym));
}

/* Unlike the symbol flattener, a tuple node is read from its AST, so a member
 * of complex type in a counterexample needs no symbol name. */
expr2tc smt_tuple_node_flattener::tuple_get(const expr2tc &expr)
{
  tuple_node_smt_astt a = to_tuple_node_ast(ctx->convert_ast(expr));
  return tuple_get_rec(a);
}

expr2tc smt_tuple_node_flattener::tuple_get_rec(tuple_node_smt_astt tuple)
{
  // XXX - what's the correct type to return here.
  std::vector<expr2tc> outmem;
  const std::vector<type2tc> &members =
    struct_union_members(tuple->sort->get_tuple_type());

  // If this tuple was free and never read, don't attempt to extract data from
  // it. There isn't any.
  if (tuple->elements.size() == 0)
  {
    for (unsigned int i = 0; i < members.size(); i++)
      outmem.emplace_back();
    return constant_struct2tc(tuple->sort->get_tuple_type(), std::move(outmem));
  }

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  for (auto const &it : members)
  {
    expr2tc res;
    if (is_tuple_ast_type(it))
    {
      res = tuple_get_rec(to_tuple_node_ast(tuple->elements[i]));
    }
    else if (is_tuple_array_ast_type(it))
    {
      res = expr2tc(); // XXX currently unimplemented
    }
    else if (is_bool_type(it) || is_number_type(it) || is_union_type(it))
    {
      res = ctx->get_by_ast(it, tuple->elements[i]);
    }
    else if (is_array_type(it) || is_vector_type(it))
    {
      // this will eventually jump to get_array()
      res = ctx->get_by_ast(it, tuple->elements[i]);
    }
    else
    {
      log_error("Unexpected type in tuple_get_rec");
      abort();
    }

    outmem.push_back(res);
    i++;
  }

  // If it's a pointer, rewrite.
  if (
    is_pointer_type(tuple->sort->get_tuple_type()) ||
    tuple->sort->get_tuple_type() == ctx->pointer_struct)
  {
    // Guard against a free pointer though
    if (is_nil_expr(outmem[0]))
      return expr2tc();

    unsigned int num = to_constant_int2t(outmem[0]).value.to_uint64();
    const BigInt &offs = to_constant_int2t(outmem[1]).value;
    return ctx->pointer_logic.back().pointer_expr(
      pointer_logict::pointert(num, offs), pointer_type2tc(get_empty_type()));
  }

  return constant_struct2tc(tuple->sort->get_tuple_type(), std::move(outmem));
}

expr2tc smt_tuple_node_flattener::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  // 2C.2d: a K>=2 struct-of-arrays value (mk_tuple_array_symbol) is a
  // tuple_node over solver-native per-field arrays, not an array_conv
  // array_ast — array_conv.get_array_elem's downcast would assert.
  // Counterexample model extraction for nested tuple-arrays is
  // unimplemented (same convention as tuple_get_rec's
  // is_tuple_array_ast_type member: return an empty expr); the verdict
  // itself comes from the solver and is unaffected.  The historical K=1
  // path passes a genuine array_conv array_ast and is unchanged.
  if (dynamic_cast<const tuple_node_smt_ast *>(array) != nullptr)
    return expr2tc();

  return array_conv.get_array_elem(
    array, index, ctx->get_flattened_array_subtype(subtype));
}

smt_astt smt_tuple_node_flattener::tuple_array_of(
  const expr2tc &init_val,
  unsigned long array_size)
{
  uint64_t elems = 1ULL << array_size;
  type2tc array_type = array_type2tc(init_val->type, gen_ulong(elems), false);
  smt_sortt array_sort = new smt_sort(
    SMT_SORT_ARRAY, array_type, array_size, ctx->convert_sort(init_val->type));

  return array_conv.convert_array_of_wsort(
    ctx->convert_ast(init_val), array_size, array_sort);
}

smt_sortt smt_tuple_node_flattener::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    unsigned int dom_width = array_domain_width_or_word_size(arrtype);

    // 2C.2a: a tuple-array sort may carry K array dimensions wrapping a
    // struct leaf (K >= 1; any dimension may be infinite).  Build the
    // range sort recursively: an inner array dimension recurses here, a
    // struct (or any non-array) leaf goes through convert_sort exactly as
    // before.  For K = 1 the subtype is the struct, the recursion branch
    // is not taken, and this is byte-identical to the historical
    //   new smt_sort(SMT_SORT_ARRAY, type, dom_width, convert_sort(subtype))
    // so every single-level array-of-struct sort is unchanged.
    smt_sortt range_sort = is_array_type(arrtype.subtype)
                             ? mk_struct_sort(arrtype.subtype)
                             : ctx->convert_sort(arrtype.subtype);

    return new smt_sort(SMT_SORT_ARRAY, type, dom_width, range_sort);
  }

  return new smt_sort(SMT_SORT_STRUCT, type);
}

void smt_tuple_node_flattener::add_tuple_constraints_for_solving()
{
  array_conv.add_array_constraints_for_solving();
}

void smt_tuple_node_flattener::push_tuple_ctx()
{
  array_conv.push_array_ctx();
}

void smt_tuple_node_flattener::note_elements_populated(
  tuple_node_smt_ast *tuple)
{
  populated_elements[ctx->ctx_level].push_back(tuple);
}

void smt_tuple_node_flattener::pop_tuple_ctx()
{
  // ctx_level has already been decremented, and smt_solver_baset::pop_ctx has
  // already deleted every AST allocated at the level being discarded. The
  // tuples registered here outlive it, so their element vectors now name
  // destroyed ASTs; drop them and let make_free rebuild on demand.
  auto it = populated_elements.find(ctx->ctx_level + 1);
  if (it != populated_elements.end())
  {
    for (tuple_node_smt_ast *tuple : it->second)
      tuple->elements.clear();
    populated_elements.erase(it);
  }

  array_conv.pop_array_ctx();
}
