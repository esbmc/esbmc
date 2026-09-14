#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_soa.h>
#include <util/expr/type_byte_size.h>
#include <util/lang/c_types.h>

/* Pointers reach the tuple interface as their synthetic (object, offset)
 * struct; everything else describes its own members. */
static type2tc struct_view(smt_solver_baset *ctx, const type2tc &t)
{
  return is_pointer_type(t) ? ctx->pointer_struct : t;
}

/** @p arrt with its innermost element type replaced by @p newelem, keeping
 *  every dimension. Used to turn "array of struct" into "array of member". */
static type2tc rebuild_array(const type2tc &arrt, const type2tc &newelem)
{
  const array_type2t &a = to_array_type(arrt);
  if (is_array_type(a.subtype))
    return array_type2tc(
      rebuild_array(a.subtype, newelem), a.array_size, a.size_is_infinite);
  return array_type2tc(newelem, a.array_size, a.size_is_infinite);
}

uint64_t smt_tuple_soa_flattener::extent(const type2tc &type) const
{
  if (!is_array_type(type))
    return 1;

  const array_type2t &a = to_array_type(type);
  if (a.size_is_infinite || is_nil_expr(a.array_size))
    return 0;
  if (!is_constant_int2t(a.array_size))
    return 0;

  return to_constant_int2t(a.array_size).value.to_uint64() * extent(a.subtype);
}

smt_sortt smt_tuple_soa_flattener::index_sort(const type2tc &arrtype) const
{
  type2tc flat = ctx->flatten_array_type(arrtype);
  return ctx->mk_int_bv_sort(
    make_array_domain_type(to_array_type(flat))->get_width());
}

smt_astt smt_tuple_soa_flattener::resize(smt_astt a, std::size_t w) const
{
  std::size_t aw = a->sort->get_data_width();
  if (aw < w)
    return ctx->mk_zero_ext(a, w - aw);
  if (aw > w)
    return ctx->mk_extract(a, w - 1, 0);
  return a;
}

smt_astt smt_tuple_soa_flattener::offset(
  smt_astt base,
  smt_astt off,
  std::size_t w) const
{
  /* The index arrives in the logical array's domain, which is narrower than
   * the flattened leaf array's whenever dimensions were collapsed. */
  off = resize(off, w);
  return base == nullptr ? off : ctx->mk_bvadd(resize(base, w), off);
}

smt_astt smt_tuple_soa_flattener::build(
  const std::string &name,
  const type2tc &type)
{
  smt_sortt s = ctx->convert_sort(type);

  if (is_array_type(type))
  {
    type2tc elem = ctx->get_flattened_array_subtype(type);

    if (is_tuple_ast_type(elem))
    {
      /* One child array per member: this is where the arrays get pushed
       * inward through the struct. */
      soa_ast *r = new soa_ast(*this, ctx, s, type);
      const type2tc view = struct_view(ctx, elem);
      const std::vector<type2tc> &members = struct_union_members(view);
      const std::vector<irep_idt> &names = struct_union_member_names(view);

      for (size_t i = 0; i < members.size(); i++)
        r->members.push_back(build(
          name + "." + names[i].as_string(), rebuild_array(type, members[i])));

      return r;
    }

    soa_ast *r = new soa_ast(*this, ctx, s, type);
    r->arr = ctx->mk_smt_symbol(
      name, ctx->mk_array_sort(index_sort(type), ctx->convert_sort(elem)));
    return r;
  }

  if (is_tuple_ast_type(type))
  {
    soa_ast *r = new soa_ast(*this, ctx, s, type);
    const type2tc view = struct_view(ctx, type);
    const std::vector<type2tc> &members = struct_union_members(view);
    const std::vector<irep_idt> &names = struct_union_member_names(view);

    for (size_t i = 0; i < members.size(); i++)
      r->members.push_back(
        build(name + "." + names[i].as_string(), members[i]));

    return r;
  }

  /* Scalars stay bare: they are handed straight to arithmetic and comparison
   * elsewhere, which would not know what to do with a wrapper. */
  return ctx->mk_smt_symbol(name, s);
}

smt_sortt smt_tuple_soa_flattener::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    unsigned int dom_width = array_domain_width_or_word_size(arrtype);
    return new smt_sort(
      SMT_SORT_ARRAY, type, dom_width, ctx->convert_sort(arrtype.subtype));
  }

  return new smt_sort(SMT_SORT_STRUCT, type);
}

smt_astt smt_tuple_soa_flattener::tuple_create(const expr2tc &structdef)
{
  const type2tc &t = structdef->type;
  soa_ast *r = new soa_ast(*this, ctx, ctx->convert_sort(t), t);

  for (size_t i = 0; i < structdef->get_num_sub_exprs(); i++)
    r->members.push_back(ctx->convert_ast(*structdef->get_sub_expr(i)));

  return r;
}

smt_astt smt_tuple_soa_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  if (name == "")
    name = ctx->mk_fresh_name("soa_fresh::");
  return build(name, s->get_tuple_type());
}

smt_astt
smt_tuple_soa_flattener::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  if (name == "NULL")
    return ctx->null_ptr_ast;

  if (name == "INVALID")
    return ctx->invalid_ptr_ast;

  assert(s->id != SMT_SORT_ARRAY);
  return build(name, s->get_tuple_type());
}

smt_astt smt_tuple_soa_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return build(sym.get_symbol_name() + "[]", sym.type);
}

/** Constrain every leaf array of @p node to hold @p value at every index. */
static void fill_const(smt_solver_baset *ctx, smt_astt node, smt_astt value)
{
  soa_astt n = to_soa_ast(node);

  if (!n->members.empty())
  {
    soa_astt v = to_soa_ast(value);
    for (size_t i = 0; i < n->members.size(); i++)
      fill_const(ctx, n->members[i], v->members[i]);
    return;
  }

  /* A leaf holds scalars, so an initialiser that is still array-shaped -- a
   * member that is itself an array, whose dimensions this leaf has absorbed --
   * has to be peeled down to its element. Every index of a constant array
   * carries the same value, so index zero is representative. */
  while (value->sort->id == SMT_SORT_ARRAY)
  {
    /* The initialiser may be one of our own leaf nodes rather than a backend
     * array -- a member that is an array of structs decomposes to leaves on
     * both sides -- so index through its flattened array, not through it. */
    if (const soa_ast *v = dynamic_cast<const soa_ast *>(value))
    {
      assert(v->members.empty() && "struct-shaped initialiser at a leaf");
      std::size_t vw = v->arr->sort->get_domain_width();
      value = ctx->mk_select(
        v->arr, v->base != nullptr ? v->base : ctx->mk_smt_bv(BigInt(0), vw));
      continue;
    }

    value = ctx->mk_select(
      value, ctx->mk_smt_bv(BigInt(0), value->sort->get_domain_width()));
  }

  /* Each leaf's domain is its own: a member that absorbed inner dimensions is
   * wider than the array being created. */
  ctx->assert_ast(n->arr->eq(
    ctx,
    ctx->array_api->convert_array_of(
      value, n->arr->sort->get_domain_width())));
}

smt_astt smt_tuple_soa_flattener::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  type2tc array_type =
    array_type2tc(init_value->type, gen_ulong(1ULL << domain_width), false);

  smt_astt fresh = build(ctx->mk_fresh_name("soa_array_of::"), array_type);
  fill_const(ctx, fresh, ctx->convert_ast(init_value));
  return fresh;
}

smt_astt smt_tuple_soa_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt)
{
  if (const_array)
  {
    smt_astt fresh =
      build(ctx->mk_fresh_name("soa_array_create::"), array_type);
    fill_const(ctx, fresh, inputargs[0]);
    return fresh;
  }

  smt_astt acc = build(ctx->mk_fresh_name("soa_array_create::"), array_type);

  const array_type2t &arr_type = to_array_type(array_type);
  if (arr_type.size_is_infinite)
    return acc;

  assert(
    is_constant_int2t(arr_type.array_size) &&
    "Non-constant sized array of type constant_array_of2t");
  uint64_t sz = to_constant_int2t(arr_type.array_size).value.to_uint64();

  for (uint64_t i = 0; i < sz; i++)
    acc = acc->update(ctx, inputargs[i], i, expr2tc());

  return acc;
}

expr2tc
smt_tuple_soa_flattener::tuple_get(const type2tc &type, smt_astt a)
{
  const type2tc view = struct_view(ctx, type);
  const std::vector<type2tc> &members = struct_union_members(view);
  soa_astt s = to_soa_ast(a);

  std::vector<expr2tc> fields;
  fields.reserve(members.size());

  for (size_t i = 0; i < members.size(); i++)
  {
    const type2tc &mt = members[i];
    smt_astt m = s->members[i];

    if (is_tuple_ast_type(mt))
      fields.push_back(tuple_get(mt, m));
    else if (is_bool_type(mt))
    {
      /* A null expr2tc is the "solver produced no value" signal (#6191). */
      tvt val = ctx->get_bool(m);
      if (val.is_unknown())
        fields.push_back(expr2tc());
      else
        fields.push_back(
          val.is_true() ? gen_true_expr() : gen_false_expr());
    }
    else if (is_bv_type(mt))
      fields.push_back(
        constant_int2tc(mt, ctx->get_bv(m, is_signedbv_type(mt))));
    else
      fields.push_back(expr2tc());
  }

  if (is_pointer_type(type))
  {
    if (is_nil_expr(fields[0]) || is_nil_expr(fields[1]))
      return expr2tc();
    pointer_logict::pointert p(
      to_constant_int2t(fields[0]).value.to_uint64(),
      to_constant_int2t(fields[1]).value);
    return ctx->pointer_logic.back().pointer_expr(p, type);
  }

  return constant_struct2tc(type, std::move(fields));
}

expr2tc smt_tuple_soa_flattener::tuple_get(const expr2tc &expr)
{
  return tuple_get(expr->type, ctx->convert_ast(expr));
}

expr2tc smt_tuple_soa_flattener::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  soa_astt a = to_soa_ast(array);
  expr2tc idx = constant_int2tc(
    make_array_domain_type(to_array_type(a->thetype)), BigInt(index));
  return tuple_get(subtype, a->select(ctx, idx));
}

smt_astt soa_ast::project(smt_solver_baset *, unsigned int elem) const
{
  assert(elem < members.size() && "Out-of-bounds tuple element accessed");
  return members[elem];
}

smt_astt soa_ast::select(smt_solver_baset *ctx, const expr2tc &idx) const
{
  assert(is_array_type(thetype) && "select on a non-array SoA ast");
  const type2tc &sub = to_array_type(thetype).subtype;

  if (!members.empty())
  {
    /* Array of structs: index each member's array, giving the struct. */
    soa_ast *r = new soa_ast(flat, ctx, ctx->convert_sort(sub), sub);
    for (smt_astt m : members)
      r->members.push_back(m->select(ctx, idx));
    return r;
  }

  smt_astt i = ctx->convert_ast(idx);

  if (is_array_type(sub))
  {
    /* A nested dimension: narrow to the sub-array that starts `idx` rows in.
     * This is the view that lets grid[i].cells[j] become one select at
     * i*extent + j rather than a shift or an ite chain. */
    uint64_t stride = flat.extent(sub);
    assert(stride != 0 && "SoA slice of an unbounded array");

    std::size_t w = arr->sort->get_domain_width();
    smt_astt off =
      ctx->mk_bvmul(flat.resize(i, w), ctx->mk_smt_bv(BigInt(stride), w));

    soa_ast *r = new soa_ast(flat, ctx, ctx->convert_sort(sub), sub);
    r->arr = arr;
    r->base = base == nullptr ? off : ctx->mk_bvadd(base, off);
    return r;
  }

  return ctx->mk_select(
    arr, flat.offset(base, i, arr->sort->get_domain_width()));
}

smt_astt soa_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  if (!is_array_type(thetype))
  {
    /* Struct field update: replace one member, share the rest. */
    soa_ast *r = new soa_ast(flat, ctx, sort, thetype);
    r->members = members;
    assert(idx < r->members.size());
    r->members[idx] = value;
    return r;
  }

  expr2tc index = is_nil_expr(idx_expr)
                    ? constant_int2tc(
                        make_array_domain_type(to_array_type(thetype)),
                        BigInt(idx))
                    : idx_expr;

  if (!members.empty())
  {
    /* Array of structs: push the store into each member's array. */
    soa_astt v = to_soa_ast(value);
    soa_ast *r = new soa_ast(flat, ctx, sort, thetype);
    for (size_t i = 0; i < members.size(); i++)
      r->members.push_back(members[i]->update(ctx, v->members[i], idx, index));
    return r;
  }

  smt_astt i = ctx->convert_ast(index);

  const type2tc &sub = to_array_type(thetype).subtype;
  if (is_array_type(sub))
  {
    soa_ast *r = new soa_ast(flat, ctx, sort, thetype);
    r->base = base;

    /* A row that came from select()ing this array already shares the leaf, so
     * the store it carries is the answer -- adopt it. */
    if (const soa_ast *v = dynamic_cast<const soa_ast *>(value))
    {
      r->arr = v->arr;
      return r;
    }

    /* A row built standalone -- a constant struct's array member, say -- has
     * its own array, and its elements have to be copied into this leaf at the
     * slice's offset. Bounded by the row length, not the array's. */
    uint64_t stride = flat.extent(sub);
    assert(stride != 0 && "SoA row copy into an unbounded array");

    /* Do the index arithmetic in the leaf array's width throughout: `i`
     * arrives in the logical array's narrower domain. */
    std::size_t w = arr->sort->get_domain_width();
    smt_astt row =
      ctx->mk_bvmul(flat.resize(i, w), ctx->mk_smt_bv(BigInt(stride), w));
    smt_astt acc = arr;
    std::size_t vw = value->sort->get_domain_width();

    for (uint64_t j = 0; j < stride; j++)
    {
      smt_astt src = ctx->mk_select(value, ctx->mk_smt_bv(BigInt(j), vw));
      smt_astt at =
        ctx->mk_bvadd(row, ctx->mk_smt_bv(BigInt(j), w));
      if (base != nullptr)
        at = ctx->mk_bvadd(flat.resize(base, w), at);
      acc = ctx->mk_store(acc, at, src);
    }

    r->arr = acc;
    return r;
  }

  soa_ast *r = new soa_ast(flat, ctx, sort, thetype);
  r->arr = ctx->mk_store(
    arr, flat.offset(base, i, arr->sort->get_domain_width()), value);
  r->base = base;
  return r;
}

smt_astt
soa_ast::eq(smt_solver_baset *ctx, smt_astt other) const
{
  soa_astt o = to_soa_ast(other);

  if (!members.empty())
  {
    smt_solver_baset::ast_vec eqs;
    eqs.reserve(members.size());
    for (size_t i = 0; i < members.size(); i++)
      eqs.push_back(members[i]->eq(ctx, o->members[i]));
    return ctx->make_n_ary_and(eqs);
  }

  return arr->eq(ctx, o->arr);
}

smt_astt
soa_ast::ite(smt_solver_baset *ctx, smt_astt cond, smt_astt falseop) const
{
  soa_astt f = to_soa_ast(falseop);
  soa_ast *r = new soa_ast(flat, ctx, sort, thetype);

  if (!members.empty())
  {
    for (size_t i = 0; i < members.size(); i++)
      r->members.push_back(members[i]->ite(ctx, cond, f->members[i]));
    return r;
  }

  r->arr = arr->ite(ctx, cond, f->arr);
  r->base = base;
  return r;
}

void soa_ast::assign(smt_solver_baset *ctx, smt_astt sym) const
{
  ctx->assert_ast(sym->eq(ctx, this));
}

void soa_ast::dump() const
{
  if (!members.empty())
    for (smt_astt m : members)
      m->dump();
  else
    arr->dump();
}
