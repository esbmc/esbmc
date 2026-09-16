#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_concat.h>
#include <util/config/config.h>
#include <util/expr/type_byte_size.h>
#include <util/lang/c_types.h>

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
smt_tuple_concat_flattener::members_of(const type2tc &type) const
{
  bool as_pointer = is_pointer_type(type) || is_code_type(type);
  return struct_union_members(as_pointer ? ctx->pointer_struct : type);
}

std::size_t smt_tuple_concat_flattener::width(const type2tc &type)
{
  auto it = width_cache.find(type);
  if (it != width_cache.end())
    return it->second;

  std::size_t w;
  if (is_bool_type(type))
    w = 1;
  else if (is_union_type(type))
    /* convert_sort already lowers a union to a bitvector of this width. */
    w = type_byte_size_bits(type, &ns).to_uint64();
  else if (is_tuple_ast_type(type))
  {
    w = 0;
    for (const type2tc &m : members_of(type))
      w += width(m);
  }
  else if (is_array_type(type))
  {
    type2tc flat = ctx->flatten_array_type(type);
    const array_type2t &a = to_array_type(flat);
    /* A flexible array member has no storage in its struct, as in C. */
    w = !is_nil_expr(a.array_size) && is_constant_int2t(a.array_size)
          ? to_constant_int2t(a.array_size).value.to_uint64() * width(a.subtype)
          : 0;
  }
  else if (is_empty_type(type))
    w = 0;
  else
    w = type->get_width();

  width_cache.emplace(type, w);
  return w;
}

std::size_t
smt_tuple_concat_flattener::offset(const type2tc &type, unsigned idx)
{
  std::vector<type2tc> ms = members_of(type);
  std::size_t off = 0;
  for (unsigned i = 0; i < idx; i++)
    off += width(ms[i]);
  return off;
}

std::size_t smt_tuple_concat_flattener::packed_width(const type2tc &type)
{
  return std::max<std::size_t>(1, width(type));
}

/** The backend array sort behind a wrapper array sort. */
static smt_sortt inner_array_sort(smt_solver_baset *ctx, smt_sortt s)
{
  return ctx->mk_array_sort(
    ctx->mk_int_bv_sort(s->get_domain_width()), s->get_range_sort());
}

/** Element @p i of the native array @p arr. */
static smt_astt
select_elem(smt_solver_baset *ctx, smt_astt arr, uint64_t i)
{
  return ctx->mk_select(
    arr, ctx->mk_smt_bv(BigInt(i), arr->sort->get_domain_width()));
}

smt_astt smt_tuple_concat_flattener::to_bv(smt_astt a, const type2tc &type)
{
  std::size_t w = width(type);
  if (w == 0)
    return ctx->mk_smt_bv(BigInt(0), 1);

  if (is_bool_type(type))
    return ctx->make_bool_bit(a);
  if (is_floatbv_type(type))
    return ctx->fp_api->mk_from_fp_to_bv(a);

  if (is_tuple_ast_type(type))
  {
    concat_smt_astt ca = to_concat_ast(a);
    if (ca->packed())
      return ca->inner;

    /* Members run from the lowest bit up; mk_concat puts its first operand
     * in the high bits. */
    std::vector<type2tc> ms = members_of(type);
    smt_astt acc = nullptr;
    for (size_t i = 0; i < ms.size(); i++)
    {
      if (width(ms[i]) == 0)
        continue;
      smt_astt m = to_bv(ca->members[i], ms[i]);
      acc = acc == nullptr ? m : ctx->mk_concat(m, acc);
    }
    return acc;
  }

  if (is_array_type(type))
  {
    /* An array member of an array element: pack it element by element. Its
     * extent is fixed, since width() is zero otherwise. */
    type2tc flat_type = ctx->flatten_array_type(type);
    const array_type2t &flat = to_array_type(flat_type);
    uint64_t n = to_constant_int2t(flat.array_size).value.to_uint64();
    bool of_structs = is_tuple_ast_type(flat.subtype);
    /* Without bools in arrays the element already is the one-bit vector. */
    bool bit_elems =
      is_bool_type(flat.subtype) && !ctx->array_api->supports_bools_in_arrays;
    smt_astt arr = of_structs ? to_concat_ast(a)->inner : a;

    smt_astt acc = nullptr;
    for (uint64_t i = 0; i < n; i++)
    {
      smt_astt e = select_elem(ctx, arr, i);
      smt_astt bits = of_structs || bit_elems ? e : to_bv(e, flat.subtype);
      acc = acc == nullptr ? bits : ctx->mk_concat(bits, acc);
    }
    return acc;
  }

  return a;
}

smt_astt smt_tuple_concat_flattener::from_bv(smt_astt raw, const type2tc &type)
{
  if (width(type) == 0)
    return build(ctx->mk_fresh_name("concat_empty::"), type);

  if (is_bool_type(type))
    return ctx->make_bit_bool(raw);
  if (is_floatbv_type(type))
    return ctx->fp_api->mk_from_bv_to_fp(raw, ctx->convert_sort(type));
  if (is_tuple_ast_type(type))
    return new concat_smt_ast(*this, ctx, ctx->convert_sort(type), type, raw);

  if (is_array_type(type))
  {
    type2tc flat_type = ctx->flatten_array_type(type);
    const array_type2t &flat = to_array_type(flat_type);
    uint64_t n = to_constant_int2t(flat.array_size).value.to_uint64();
    std::size_t ew = packed_width(flat.subtype);
    bool of_structs = is_tuple_ast_type(flat.subtype);
    bool bit_elems =
      is_bool_type(flat.subtype) && !ctx->array_api->supports_bools_in_arrays;

    smt_sortt s = ctx->convert_sort(flat_type);
    smt_astt arr = ctx->mk_smt_symbol(
      ctx->mk_fresh_name("concat_member_array::"),
      of_structs ? inner_array_sort(ctx, s) : s);

    for (uint64_t i = 0; i < n; i++)
    {
      smt_astt bits = ctx->mk_extract(raw, (i + 1) * ew - 1, i * ew);
      smt_astt v =
        of_structs || bit_elems ? bits : from_bv(bits, flat.subtype);
      arr = ctx->mk_store(
        arr, ctx->mk_smt_bv(BigInt(i), arr->sort->get_domain_width()), v);
    }

    if (of_structs)
      return new concat_smt_ast(*this, ctx, s, flat_type, arr);
    return arr;
  }

  return raw;
}

smt_astt
smt_tuple_concat_flattener::build(const std::string &name, const type2tc &type)
{
  smt_sortt s = ctx->convert_sort(type);

  if (is_tuple_array_ast_type(type))
    return new concat_smt_ast(
      *this, ctx, s, type, ctx->mk_smt_symbol(name, inner_array_sort(ctx, s)));

  if (!is_tuple_ast_type(type))
    return ctx->mk_smt_symbol(name, s);

  std::vector<type2tc> ms = members_of(type);
  std::vector<irep_idt> names = struct_union_member_names(
    is_pointer_type(type) || is_code_type(type) ? ctx->pointer_struct : type);

  std::vector<smt_astt> terms;
  terms.reserve(ms.size());
  for (size_t i = 0; i < ms.size(); i++)
    terms.push_back(build(name + "." + names[i].as_string(), ms[i]));

  return new concat_smt_ast(*this, ctx, s, type, std::move(terms));
}

smt_sortt smt_tuple_concat_flattener::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    type2tc t = unrewrite(ctx, type);
    const array_type2t &arrtype = to_array_type(t);
    assert(
      !is_array_type(arrtype.subtype) &&
      "Array dimensions should be flattened before the tuple interface");
    return new smt_sort(
      SMT_SORT_ARRAY,
      t,
      array_domain_width_or_word_size(arrtype),
      ctx->mk_int_bv_sort(packed_width(arrtype.subtype)));
  }

  return new smt_sort(SMT_SORT_STRUCT, type);
}

smt_astt smt_tuple_concat_flattener::tuple_create(const expr2tc &structdef)
{
  std::vector<smt_astt> terms;
  terms.reserve(structdef->get_num_sub_exprs());
  for (size_t i = 0; i < structdef->get_num_sub_exprs(); i++)
    terms.push_back(ctx->convert_ast(*structdef->get_sub_expr(i)));

  const type2tc &t = structdef->type;
  return new concat_smt_ast(
    *this, ctx, ctx->convert_sort(t), t, std::move(terms));
}

smt_astt smt_tuple_concat_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  if (name == "")
    name = ctx->mk_fresh_name("concat_fresh::");
  return build(name, s->get_tuple_type());
}

smt_astt smt_tuple_concat_flattener::mk_tuple_symbol(
  const std::string &name,
  smt_sortt s)
{
  if (name == "NULL")
    return ctx->null_ptr_ast;

  if (name == "INVALID")
    return ctx->invalid_ptr_ast;

  assert(s->id != SMT_SORT_ARRAY);
  return build(name, s->get_tuple_type());
}

smt_astt smt_tuple_concat_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  return build(sym.get_symbol_name(), ctx->flatten_array_type(sym.type));
}

smt_astt smt_tuple_concat_flattener::tuple_array_of(
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
  smt_astt elem = to_bv(ctx->convert_ast(init_value), init_value->type);
  return new concat_smt_ast(
    *this,
    ctx,
    ctx->convert_sort(array_type),
    array_type,
    ctx->array_api->convert_array_of(elem, domain_width));
}

smt_astt smt_tuple_concat_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain)
{
  type2tc type = unrewrite(ctx, array_type);
  smt_sortt s = ctx->convert_sort(type);
  const array_type2t &arr_type = to_array_type(type);

  if (const_array)
    return new concat_smt_ast(
      *this,
      ctx,
      s,
      type,
      ctx->array_api->convert_array_of(
        to_bv(inputargs[0], arr_type.subtype), domain->get_data_width()));

  smt_astt acc = ctx->mk_smt_symbol(
    ctx->mk_fresh_name("concat_array_create::"), inner_array_sort(ctx, s));

  if (!arr_type.size_is_infinite)
  {
    assert(
      is_constant_int2t(arr_type.array_size) &&
      "Non-constant sized array of type constant_array_of2t");
    uint64_t sz = to_constant_int2t(arr_type.array_size).value.to_uint64();
    for (uint64_t i = 0; i < sz; i++)
      acc = ctx->mk_store(
        acc,
        ctx->mk_smt_bv(BigInt(i), s->get_domain_width()),
        to_bv(inputargs[i], arr_type.subtype));
  }

  return new concat_smt_ast(*this, ctx, s, type, acc);
}

expr2tc
smt_tuple_concat_flattener::tuple_get(const type2tc &type, smt_astt a)
{
  concat_smt_astt ca = to_concat_ast(a);
  std::vector<type2tc> ms = members_of(type);

  std::vector<expr2tc> fields;
  fields.reserve(ms.size());
  for (size_t i = 0; i < ms.size(); i++)
  {
    smt_astt m = ca->packed() ? ca->project(ctx, i) : ca->members[i];
    fields.push_back(
      is_tuple_ast_type(ms[i]) ? tuple_get(ms[i], m)
                               : ctx->get_by_ast(ms[i], m));
  }

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

expr2tc smt_tuple_concat_flattener::tuple_get(const expr2tc &expr)
{
  return tuple_get(expr->type, ctx->convert_ast(expr));
}

expr2tc smt_tuple_concat_flattener::tuple_get_array_elem(
  smt_astt array,
  uint64_t index,
  const type2tc &subtype)
{
  smt_astt elem = select_elem(ctx, to_concat_ast(array)->inner, index);
  return tuple_get(
    subtype,
    new concat_smt_ast(*this, ctx, ctx->convert_sort(subtype), subtype, elem));
}

smt_astt concat_smt_ast::ite(
  smt_solver_baset *ctx,
  smt_astt cond,
  smt_astt falseop) const
{
  concat_smt_astt f = to_concat_ast(falseop);

  /* An array of structs is already a native array: choose between the arrays,
   * not between their serialisations. */
  if (is_array_type(thetype))
    return new concat_smt_ast(
      flat, ctx, sort, thetype, ctx->mk_ite(cond, inner, f->inner));

  if (!packed() && !f->packed())
  {
    std::vector<smt_astt> terms;
    terms.reserve(members.size());
    for (size_t i = 0; i < members.size(); i++)
      terms.push_back(members[i]->ite(ctx, cond, f->members[i]));
    return new concat_smt_ast(flat, ctx, sort, thetype, std::move(terms));
  }

  return new concat_smt_ast(
    flat,
    ctx,
    sort,
    thetype,
    ctx->mk_ite(cond, flat.to_bv(this, thetype), flat.to_bv(f, thetype)));
}

smt_astt concat_smt_ast::eq(smt_solver_baset *ctx, smt_astt other) const
{
  concat_smt_astt o = to_concat_ast(other);

  if (is_array_type(thetype))
    return ctx->mk_eq(inner, o->inner);

  if (!packed() && !o->packed())
  {
    std::vector<type2tc> ms = flat.members_of(thetype);
    smt_solver_baset::ast_vec eqs;
    for (size_t i = 0; i < ms.size(); i++)
      if (flat.width(ms[i]) != 0)
        eqs.push_back(members[i]->eq(ctx, o->members[i]));
    return ctx->make_n_ary_and(eqs);
  }

  return ctx->mk_eq(flat.to_bv(this, thetype), flat.to_bv(o, thetype));
}

smt_astt concat_smt_ast::select(smt_solver_baset *ctx, const expr2tc &idx) const
{
  assert(is_array_type(thetype) && "select on a non-array concat ast");
  const type2tc &subtype = to_array_type(thetype).subtype;
  return new concat_smt_ast(
    flat,
    ctx,
    ctx->convert_sort(subtype),
    subtype,
    ctx->mk_select(inner, ctx->convert_ast(idx)));
}

smt_astt concat_smt_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  if (is_array_type(thetype))
  {
    const type2tc &subtype = to_array_type(thetype).subtype;
    smt_astt i = is_nil_expr(idx_expr)
                   ? ctx->mk_smt_bv(BigInt(idx), sort->get_domain_width())
                   : ctx->convert_ast(idx_expr);
    return new concat_smt_ast(
      flat,
      ctx,
      sort,
      thetype,
      ctx->mk_store(inner, i, flat.to_bv(value, subtype)));
  }

  std::vector<type2tc> ms = flat.members_of(thetype);
  assert(idx < ms.size());

  if (!packed())
  {
    std::vector<smt_astt> terms = members;
    terms[idx] = value;
    return new concat_smt_ast(flat, ctx, sort, thetype, std::move(terms));
  }

  /* Splice the new field into the word: the bits above it, the new value,
   * the bits below it. mk_concat puts its first operand high. */
  std::size_t w = flat.width(ms[idx]);
  if (w == 0)
    return this;

  std::size_t total = flat.width(thetype);
  std::size_t off = flat.offset(thetype, idx);

  smt_astt result = flat.to_bv(value, ms[idx]);
  if (off > 0)
    result = ctx->mk_concat(result, ctx->mk_extract(inner, off - 1, 0));
  if (off + w < total)
    result = ctx->mk_concat(ctx->mk_extract(inner, total - 1, off + w), result);

  return new concat_smt_ast(flat, ctx, sort, thetype, result);
}

smt_astt concat_smt_ast::project(smt_solver_baset *ctx, unsigned int elem) const
{
  assert(!is_array_type(thetype) && "project on a concat array ast");

  if (!packed())
  {
    assert(elem < members.size() && "Out-of-bounds tuple element accessed");
    return members[elem];
  }

  const type2tc m = flat.members_of(thetype)[elem];
  std::size_t w = flat.width(m);
  if (w == 0)
    return flat.from_bv(inner, m);

  std::size_t off = flat.offset(thetype, elem);
  return flat.from_bv(ctx->mk_extract(inner, off + w - 1, off), m);
}

void concat_smt_ast::dump() const
{
  if (packed())
    inner->dump();
  else
    for (smt_astt m : members)
      m->dump();
}
