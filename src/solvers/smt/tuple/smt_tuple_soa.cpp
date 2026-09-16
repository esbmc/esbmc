#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_soa.h>
#include <util/config/config.h>
#include <util/expr/type_byte_size.h>
#include <util/lang/c_types.h>

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

/* convert_sort and tuple_array_create_despatch hand array-of-struct types over
 * with every pointer rewritten to pointer_struct -- pointers inside a union
 * too, which widens the union. Terms built elsewhere from the untouched type
 * keep the C width, so undo the rewrite where types enter this flattener. */
static type2tc unrewrite(smt_solver_baset *ctx, type2tc type)
{
  struct
  {
    const type2tc &pointer_struct;

    void operator()(type2tc &e) const
    {
      if (e == pointer_struct)
        e = pointer_type2tc(get_empty_type());
      else
        e->Foreach_subtype(*this);
    }
  } delegate = {ctx->pointer_struct};

  type->Foreach_subtype(delegate);
  return type;
}

std::vector<type2tc>
smt_tuple_soa_flattener::members_of(const type2tc &type) const
{
  bool as_pointer = is_pointer_type(type) || is_code_type(type);
  return struct_union_members(as_pointer ? ctx->pointer_struct : type);
}

uint64_t smt_tuple_soa_flattener::extent(const type2tc &type) const
{
  if (!is_array_type(type))
    return 1;

  const array_type2t &a = to_array_type(type);
  if (
    a.size_is_infinite || is_nil_expr(a.array_size) ||
    !is_constant_int2t(a.array_size))
    return 0;

  return to_constant_int2t(a.array_size).value.to_uint64() * extent(a.subtype);
}

smt_sortt smt_tuple_soa_flattener::flat_sort(const type2tc &type) const
{
  type2tc flat = ctx->flatten_array_type(type);
  return ctx->mk_array_sort(
    ctx->mk_int_bv_sort(
      make_array_domain_type(to_array_type(flat))->get_width()),
    ctx->convert_sort(ctx->get_flattened_array_subtype(type)));
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

smt_astt smt_tuple_soa_flattener::row(
  smt_astt arr,
  smt_astt start,
  const type2tc &rowtype)
{
  smt_sortt s = flat_sort(rowtype);
  smt_astt out = ctx->mk_smt_symbol(ctx->mk_fresh_name("soa_row::"), s);

  uint64_t n = extent(rowtype);
  assert(n != 0 && "SoA row of an array without a constant size");
  std::size_t w = arr->sort->get_domain_width();
  for (uint64_t j = 0; j < n; j++)
    out = ctx->mk_store(
      out,
      ctx->mk_smt_bv(BigInt(j), s->get_domain_width()),
      ctx->mk_select(arr, ctx->mk_bvadd(start, ctx->mk_smt_bv(BigInt(j), w))));

  return out;
}

smt_astt smt_tuple_soa_flattener::build(
  const std::string &name,
  const type2tc &type,
  bool in_node)
{
  smt_sortt s = ctx->convert_sort(type);

  if (is_array_type(type))
  {
    type2tc elem = ctx->get_flattened_array_subtype(type);

    if (is_tuple_ast_type(elem))
    {
      /* One array per member: this is where arrays are pushed inward through
       * the struct. */
      soa_ast *r = new soa_ast(*this, ctx, s, type);
      std::vector<type2tc> ms = members_of(elem);
      std::vector<irep_idt> names = struct_union_member_names(
        is_pointer_type(elem) || is_code_type(elem) ? ctx->pointer_struct
                                                    : elem);
      for (size_t i = 0; i < ms.size(); i++)
        r->members.push_back(build(
          name + "." + names[i].as_string(), rebuild_array(type, ms[i]), true));
      return r;
    }

    if (in_node && is_array_type(to_array_type(type).subtype))
    {
      soa_ast *r = new soa_ast(*this, ctx, s, type);
      r->arr = ctx->mk_smt_symbol(name, flat_sort(type));
      return r;
    }

    return ctx->mk_smt_symbol(name, s);
  }

  if (is_tuple_ast_type(type))
  {
    soa_ast *r = new soa_ast(*this, ctx, s, type);
    std::vector<type2tc> ms = members_of(type);
    std::vector<irep_idt> names = struct_union_member_names(
      is_pointer_type(type) || is_code_type(type) ? ctx->pointer_struct : type);
    for (size_t i = 0; i < ms.size(); i++)
      r->members.push_back(
        build(name + "." + names[i].as_string(), ms[i], false));
    return r;
  }

  return ctx->mk_smt_symbol(name, s);
}

smt_sortt smt_tuple_soa_flattener::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    type2tc t = unrewrite(ctx, type);
    const array_type2t &arrtype = to_array_type(t);
    return new smt_sort(
      SMT_SORT_ARRAY,
      t,
      array_domain_width_or_word_size(arrtype),
      ctx->convert_sort(arrtype.subtype));
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
  return build(name, s->get_tuple_type(), false);
}

smt_astt
smt_tuple_soa_flattener::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  if (name == "NULL")
    return ctx->null_ptr_ast;

  if (name == "INVALID")
    return ctx->invalid_ptr_ast;

  assert(s->id != SMT_SORT_ARRAY);
  return build(name, s->get_tuple_type(), false);
}

smt_astt smt_tuple_soa_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return build(sym.get_symbol_name() + "[]", sym.type, false);
}

void smt_tuple_soa_flattener::fill_const(
  smt_astt node,
  smt_astt value,
  const type2tc &type)
{
  type2tc elem = ctx->get_flattened_array_subtype(type);

  if (is_tuple_ast_type(elem))
  {
    soa_astt n = to_soa_ast(node);
    soa_astt v = to_soa_ast(value);
    std::vector<type2tc> ms = members_of(elem);
    for (size_t i = 0; i < ms.size(); i++)
      fill_const(n->members[i], v->members[i], rebuild_array(type, ms[i]));
    return;
  }

  /* The initialiser of a member that is itself an array is array-shaped. Every
   * index of a constant array holds the same value, so index zero stands for
   * all of them. */
  while (value->sort->id == SMT_SORT_ARRAY)
  {
    const soa_ast *l = dynamic_cast<const soa_ast *>(value);
    smt_astt a = l != nullptr ? l->arr : value;
    value = ctx->mk_select(
      a, ctx->mk_smt_bv(BigInt(0), a->sort->get_domain_width()));
  }

  const soa_ast *l = dynamic_cast<const soa_ast *>(node);
  smt_astt target = l != nullptr ? l->arr : node;
  ctx->assert_ast(ctx->mk_eq(
    target,
    ctx->array_api->convert_array_of(
      value, target->sort->get_domain_width())));
}

smt_astt smt_tuple_soa_flattener::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  /* The caller passes the real array's domain width. An array of 2^(dw-1)
   * elements is one ESBMC gives exactly that width (size_to_bit_width); at the
   * word size the real array is one without a constant size. */
  type2tc array_type =
    domain_width >= config.ansi_c.word_size
      ? array_type2tc(init_value->type, expr2tc(), true)
      : array_type2tc(
          init_value->type, gen_ulong(1ULL << (domain_width - 1)), false);

  smt_astt fresh =
    build(ctx->mk_fresh_name("soa_array_of::"), array_type, false);
  fill_const(fresh, ctx->convert_ast(init_value), array_type);
  return fresh;
}

smt_astt smt_tuple_soa_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt)
{
  type2tc type = unrewrite(ctx, array_type);
  smt_astt acc = build(ctx->mk_fresh_name("soa_array_create::"), type, false);

  if (const_array)
  {
    fill_const(acc, inputargs[0], type);
    return acc;
  }

  const array_type2t &arr_type = to_array_type(type);
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
  std::vector<type2tc> ms = members_of(type);
  soa_astt s = to_soa_ast(a);

  std::vector<expr2tc> fields;
  fields.reserve(ms.size());
  for (size_t i = 0; i < ms.size(); i++)
    fields.push_back(
      is_tuple_ast_type(ms[i]) ? tuple_get(ms[i], s->members[i])
                               : ctx->get_by_ast(ms[i], s->members[i]));

  if (is_pointer_type(type) || is_code_type(type))
  {
    if (is_nil_expr(fields[0]) || is_nil_expr(fields[1]))
      return expr2tc();
    return ctx->pointer_logic.back().pointer_expr(
      pointer_logict::pointert(
        to_constant_int2t(fields[0]).value.to_uint64(),
        to_constant_int2t(fields[1]).value),
      type);
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
  assert(!leaf() && elem < members.size() && "Bad tuple element accessed");
  return members[elem];
}

smt_astt soa_ast::select(smt_solver_baset *ctx, const expr2tc &idx) const
{
  assert(is_array_type(thetype) && "select on a non-array SoA ast");
  const type2tc &sub = to_array_type(thetype).subtype;

  if (!leaf())
  {
    soa_ast *r = new soa_ast(flat, ctx, ctx->convert_sort(sub), sub);
    for (smt_astt m : members)
      r->members.push_back(m->select(ctx, idx));
    return r;
  }

  std::size_t w = arr->sort->get_domain_width();
  smt_astt start = ctx->mk_bvmul(
    flat.resize(ctx->convert_ast(idx), w),
    ctx->mk_smt_bv(BigInt(flat.extent(sub)), w));
  return flat.row(arr, start, sub);
}

smt_astt soa_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  if (!is_array_type(thetype))
  {
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

  soa_ast *r = new soa_ast(flat, ctx, sort, thetype);

  if (!leaf())
  {
    soa_astt v = to_soa_ast(value);
    for (size_t i = 0; i < members.size(); i++)
      r->members.push_back(members[i]->update(ctx, v->members[i], idx, index));
    return r;
  }

  /* Storing a whole row, a backend array: copy its slots into place. */
  const type2tc &sub = to_array_type(thetype).subtype;
  uint64_t n = flat.extent(sub);
  assert(n != 0 && "SoA row store into an array without a constant size");

  std::size_t w = arr->sort->get_domain_width();
  std::size_t vw = value->sort->get_domain_width();
  smt_astt start = ctx->mk_bvmul(
    flat.resize(ctx->convert_ast(index), w), ctx->mk_smt_bv(BigInt(n), w));

  r->arr = arr;
  for (uint64_t j = 0; j < n; j++)
    r->arr = ctx->mk_store(
      r->arr,
      ctx->mk_bvadd(start, ctx->mk_smt_bv(BigInt(j), w)),
      ctx->mk_select(value, ctx->mk_smt_bv(BigInt(j), vw)));
  return r;
}

smt_astt soa_ast::eq(smt_solver_baset *ctx, smt_astt other) const
{
  soa_astt o = to_soa_ast(other);

  if (leaf())
    return ctx->mk_eq(arr, o->arr);

  smt_solver_baset::ast_vec eqs;
  eqs.reserve(members.size());
  for (size_t i = 0; i < members.size(); i++)
    eqs.push_back(members[i]->eq(ctx, o->members[i]));
  return ctx->make_n_ary_and(eqs);
}

smt_astt
soa_ast::ite(smt_solver_baset *ctx, smt_astt cond, smt_astt falseop) const
{
  soa_astt f = to_soa_ast(falseop);
  soa_ast *r = new soa_ast(flat, ctx, sort, thetype);

  if (leaf())
  {
    r->arr = ctx->mk_ite(cond, arr, f->arr);
    return r;
  }

  for (size_t i = 0; i < members.size(); i++)
    r->members.push_back(members[i]->ite(ctx, cond, f->members[i]));
  return r;
}

void soa_ast::assign(smt_solver_baset *ctx, smt_astt sym) const
{
  ctx->assert_ast(eq(ctx, sym));
}

void soa_ast::dump() const
{
  if (leaf())
    arr->dump();
  else
    for (smt_astt m : members)
      m->dump();
}
