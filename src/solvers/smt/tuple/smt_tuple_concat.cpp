#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <solvers/smt/tuple/smt_tuple_concat.h>
#include <util/expr/type_byte_size.h>
#include <util/lang/c_types.h>

/* Pointers reach the tuple interface as their synthetic (object, offset)
 * struct; everything else describes its own members. */
static type2tc struct_view(smt_solver_baset *ctx, const type2tc &t)
{
  return is_pointer_type(t) ? ctx->pointer_struct : t;
}

std::size_t smt_tuple_concat_flattener::bv_width(const type2tc &type) const
{
  return type_byte_size_bits(type, &ns).to_uint64();
}

smt_sortt smt_tuple_concat_flattener::bv_sort(const type2tc &type) const
{
  return ctx->mk_int_bv_sort(bv_width(type));
}

smt_astt smt_tuple_concat_flattener::wrap(smt_astt raw, const type2tc &type)
{
  return new concat_smt_ast(*this, ctx, ctx->convert_sort(type), raw, type);
}

smt_astt smt_tuple_concat_flattener::to_bv(smt_astt a, const type2tc &t)
{
  if (is_tuple_ast_type(t))
    return to_concat_ast(a)->inner;
  if (is_bool_type(t))
    return ctx->make_bool_bit(a);
  if (is_floatbv_type(t))
    return ctx->fp_api->mk_from_fp_to_bv(a);
  return a;
}

smt_astt smt_tuple_concat_flattener::from_bv(smt_astt raw, const type2tc &t)
{
  if (is_tuple_ast_type(t))
    return wrap(raw, t);
  if (is_bool_type(t))
    return ctx->make_bit_bool(raw);
  if (is_floatbv_type(t))
    return ctx->fp_api->mk_from_bv_to_fp(raw, ctx->convert_sort(t));
  if (is_array_type(t))
  {
    /* An array member sits at a symbolic bit offset once indexed, which SMT
     * extract cannot express (its indices are numerals). Emulating it costs a
     * variable shift over the whole struct, which is the case this encoding
     * is worst at -- see #37. Refuse rather than encode it badly. */
    log_error("--tuple-concat-flattener: arrays inside structs unsupported");
    abort();
  }
  return raw;
}

smt_sortt smt_tuple_concat_flattener::mk_struct_sort(const type2tc &type)
{
  if (is_array_type(type))
  {
    const array_type2t &arrtype = to_array_type(type);
    assert(
      !is_array_type(arrtype.subtype) &&
      "Array dimensions should be flattened before the tuple interface");
    unsigned int dom_width = array_domain_width_or_word_size(arrtype);
    return new smt_sort(
      SMT_SORT_ARRAY, type, dom_width, bv_sort(arrtype.subtype));
  }

  return new smt_sort(SMT_SORT_STRUCT, type);
}

/** The backend array sort behind a wrapper array sort. */
static smt_sortt inner_array_sort(smt_solver_baset *ctx, smt_sortt s)
{
  return ctx->mk_array_sort(
    ctx->mk_int_bv_sort(s->get_domain_width()), s->get_range_sort());
}

smt_astt smt_tuple_concat_flattener::tuple_create(const expr2tc &structdef)
{
  /* Concatenate the members here rather than via bitcast2tc: convert_bitcast
   * reaches the struct's members with member2tc, whose conversion converts the
   * struct again, and the recursion does not terminate. */
  const type2tc &t = structdef->type;
  const type2tc view = struct_view(ctx, t);
  const std::vector<type2tc> &members = struct_union_members(view);
  assert(structdef->get_num_sub_exprs() == members.size());

  smt_astt acc = nullptr;
  for (size_t i = 0; i < members.size(); i++)
  {
    if (bv_width(members[i]) == 0)
      continue;

    smt_astt m =
      to_bv(ctx->convert_ast(*structdef->get_sub_expr(i)), members[i]);

    /* Members run from the lowest bit up, and mk_concat puts its first
     * operand in the high bits. */
    acc = acc == nullptr ? m : ctx->mk_concat(m, acc);
  }

  assert(acc != nullptr && "Wholly zero-width struct reached tuple_create");
  return wrap(acc, t);
}

smt_astt smt_tuple_concat_flattener::tuple_fresh(smt_sortt s, std::string name)
{
  if (name == "")
    name = ctx->mk_fresh_name("concat_fresh::");

  if (s->id == SMT_SORT_ARRAY)
    return new concat_smt_ast(
      *this,
      ctx,
      s,
      ctx->mk_smt_symbol(name, inner_array_sort(ctx, s)),
      s->get_tuple_type());

  const type2tc &t = s->get_tuple_type();
  return new concat_smt_ast(
    *this, ctx, s, ctx->mk_smt_symbol(name, bv_sort(t)), t);
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
  const type2tc &t = s->get_tuple_type();
  return new concat_smt_ast(
    *this, ctx, s, ctx->mk_smt_symbol(name, bv_sort(t)), t);
}

smt_astt smt_tuple_concat_flattener::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  type2tc flat_type = ctx->flatten_array_type(sym.type);
  smt_sortt s = ctx->convert_sort(flat_type);
  return new concat_smt_ast(
    *this,
    ctx,
    s,
    ctx->mk_smt_symbol(sym.get_symbol_name(), inner_array_sort(ctx, s)),
    flat_type);
}

smt_astt smt_tuple_concat_flattener::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  smt_astt elem = to_bv(ctx->convert_ast(init_value), init_value->type);
  type2tc array_type = array_type2tc(
    init_value->type, gen_ulong(1ULL << domain_width), false);
  return new concat_smt_ast(
    *this,
    ctx,
    ctx->convert_sort(array_type),
    ctx->array_api->convert_array_of(elem, domain_width),
    array_type);
}

smt_astt smt_tuple_concat_flattener::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain)
{
  smt_sortt s = ctx->convert_sort(array_type);
  const array_type2t &arr_type = to_array_type(array_type);

  if (const_array)
    return new concat_smt_ast(
      *this,
      ctx,
      s,
      ctx->array_api->convert_array_of(
        to_bv(inputargs[0], arr_type.subtype), domain->get_data_width()),
      array_type);

  smt_astt acc = ctx->mk_smt_symbol(
    ctx->mk_fresh_name("concat_array_create::"), inner_array_sort(ctx, s));

  if (arr_type.size_is_infinite)
    return new concat_smt_ast(*this, ctx, s, acc, array_type);

  assert(
    is_constant_int2t(arr_type.array_size) &&
    "Non-constant sized array of type constant_array_of2t");
  uint64_t sz = to_constant_int2t(arr_type.array_size).value.to_uint64();

  for (uint64_t i = 0; i < sz; i++)
    acc = ctx->mk_store(
      acc,
      ctx->mk_smt_bv(BigInt(i), s->get_domain_width()),
      to_bv(inputargs[i], arr_type.subtype));

  return new concat_smt_ast(*this, ctx, s, acc, array_type);
}

/* BigInt has no shift or mask operators, so slice with divide and modulo. */
static BigInt two_pow(std::size_t n)
{
  BigInt r(1);
  for (std::size_t i = 0; i < n; i++)
    r *= 2;
  return r;
}

/** Decompose the model value of a struct-shaped bitvector into its members. */
static expr2tc decode(
  smt_tuple_concat_flattener &flat,
  const BigInt &val,
  const type2tc &type,
  const namespacet &ns)
{
  const type2tc view = struct_view(flat.ctx, type);
  const std::vector<type2tc> &members = struct_union_members(view);
  const std::vector<irep_idt> &names = struct_union_member_names(view);

  std::vector<expr2tc> fields;
  fields.reserve(members.size());

  for (size_t i = 0; i < members.size(); i++)
  {
    std::size_t w = flat.bv_width(members[i]);
    if (w == 0)
    {
      fields.push_back(gen_zero(members[i]));
      continue;
    }

    std::size_t off = member_offset_bits(view, names[i], &ns).to_uint64();
    BigInt modulus = two_pow(w);
    BigInt field = (val / two_pow(off)) % modulus;

    if (is_tuple_ast_type(members[i]))
      fields.push_back(decode(flat, field, members[i], ns));
    else if (is_bool_type(members[i]))
      fields.push_back(field == 0 ? gen_false_expr() : gen_true_expr());
    else if (is_bv_type(members[i]))
    {
      /* The slice is unsigned; a signed member whose sign bit is set names
       * the negative value that far below the wrap-around point. */
      if (is_signedbv_type(members[i]) && field >= two_pow(w - 1))
        field -= modulus;
      fields.push_back(constant_int2tc(members[i], field));
    }
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
    return flat.ctx->pointer_logic.back().pointer_expr(p, type);
  }

  return constant_struct2tc(type, std::move(fields));
}

expr2tc
smt_tuple_concat_flattener::tuple_get(const type2tc &type, smt_astt a)
{
  return decode(*this, ctx->get_bv(to_concat_ast(a)->inner, false), type, ns);
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
  concat_smt_astt ca = to_concat_ast(array);
  smt_astt elem = ctx->mk_select(
    ca->inner, ctx->mk_smt_bv(BigInt(index), ca->sort->get_domain_width()));
  return decode(*this, ctx->get_bv(elem, false), subtype, ns);
}

smt_astt concat_smt_ast::ite(
  smt_solver_baset *ctx,
  smt_astt cond,
  smt_astt falseop) const
{
  return new concat_smt_ast(
    flat,
    ctx,
    sort,
    ctx->mk_ite(cond, inner, to_concat_ast(falseop)->inner),
    thetype);
}

smt_astt concat_smt_ast::eq(smt_solver_baset *ctx, smt_astt other) const
{
  return ctx->mk_eq(inner, to_concat_ast(other)->inner);
}

smt_astt concat_smt_ast::select(smt_solver_baset *ctx, const expr2tc &idx) const
{
  assert(is_array_type(thetype) && "select on a non-array concat ast");
  return flat.wrap(
    ctx->mk_select(inner, ctx->convert_ast(idx)),
    to_array_type(thetype).subtype);
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
      ctx->mk_store(inner, i, flat.to_bv(value, subtype)),
      thetype);
  }

  /* Splice the new field into the bitvector: the bits above it, the new
   * value, the bits below it. mk_concat puts its first operand high. */
  const type2tc view = struct_view(ctx, thetype);
  const std::vector<type2tc> &members = struct_union_members(view);
  const std::vector<irep_idt> &names = struct_union_member_names(view);
  assert(idx < members.size());

  std::size_t total = flat.bv_width(thetype);
  std::size_t w = flat.bv_width(members[idx]);
  if (w == 0)
    return this;

  std::size_t off = member_offset_bits(view, names[idx], &flat.ns).to_uint64();

  smt_astt result = flat.to_bv(value, members[idx]);
  if (off > 0)
    result = ctx->mk_concat(result, ctx->mk_extract(inner, off - 1, 0));
  if (off + w < total)
    result = ctx->mk_concat(ctx->mk_extract(inner, total - 1, off + w), result);

  return new concat_smt_ast(flat, ctx, sort, result, thetype);
}

smt_astt concat_smt_ast::project(smt_solver_baset *ctx, unsigned int elem) const
{
  assert(!is_array_type(thetype) && "project on a concat array ast");

  const type2tc view = struct_view(ctx, thetype);
  const std::vector<type2tc> &members = struct_union_members(view);
  const std::vector<irep_idt> &names = struct_union_member_names(view);
  assert(elem < members.size() && "Out-of-bounds tuple element accessed");

  std::size_t w = flat.bv_width(members[elem]);
  if (w == 0)
    return ctx->convert_ast(gen_zero(members[elem]));

  std::size_t off = member_offset_bits(view, names[elem], &flat.ns).to_uint64();
  return flat.from_bv(ctx->mk_extract(inner, off + w - 1, off), members[elem]);
}
