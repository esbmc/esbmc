#include <irep2/irep2_utils.h>
#include <irep2/irep2_type.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/std_types.h>
#include <util/type_byte_size.h>

namespace
{
struct type_sizet
{
  const namespacet *ns;

  explicit type_sizet(const namespacet *ns) : ns(ns)
  {
  }
  type_sizet(const type_sizet &) = delete;
  type_sizet &operator=(const type_sizet &) = delete;

  BigInt member_offset_bits(const type2tc &type, const irep_idt &member) const;
  BigInt size_bits(const type2tc &type) const;
  expr2tc size_bits_expr(const type2tc &type) const;
  expr2tc pointer_offset_bits(const expr2tc &expr) const;
};
} // namespace

BigInt type_sizet::member_offset_bits(
  const type2tc &type,
  const irep_idt &member) const
{
  BigInt result = 0;

  // empty union generate an array
  if (!is_struct_type(type))
    return result;

  unsigned idx = 0;
  const struct_type2t &thetype = to_struct_type(type);
  for (auto const &it : thetype.members)
  {
    if (thetype.member_names[idx] == member.as_string())
      break;

    result += size_bits(it);
    idx++;
  }

  return result;
}

BigInt member_offset_bits(
  const type2tc &type,
  const irep_idt &member,
  const namespacet *ns)
{
  return type_sizet(ns).member_offset_bits(type, member);
}

BigInt
member_offset(const type2tc &type, const irep_idt &member, const namespacet *ns)
{
  return member_offset_bits(type, member, ns) / 8;
}

BigInt type_byte_size_default(
  const type2tc &type,
  const BigInt &defaultval,
  const namespacet *ns)
{
  try
  {
    return type_byte_size(type, ns);
  }
  catch (const array_type2t::dyn_sized_array_excp &e)
  {
    return defaultval;
  }
}

BigInt type_sizet::size_bits(const type2tc &type) const
{
  switch (type->type_id)
  {
  // This is a gcc extension.
  // https://gcc.gnu.org/onlinedocs/gcc-4.8.0/gcc/Pointer-Arith.html
  case type2t::empty_id:
    return 1;

  case type2t::code_id:
    return 0;

  case type2t::symbol_id:
    if (ns)
      return size_bits(ns->follow(type));
    log_error("Symbolic type id in size_typet::size_bits\n{}", *type);
    abort();

  case type2t::cpp_name_id:
    if (ns)
      return size_bits(ns->follow(type));
    log_error("Symbolic C++ type id in size_typet::size_bits\n{}", *type);
    abort();

  case type2t::bool_id:
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
  case type2t::pointer_id:
    return type->get_width();

  case type2t::vector_id:
  case type2t::array_id:
  {
    // Attempt to compute constant array offset. If we can't, we can't
    // reasonably return anything anyway, so throw.
    const array_data &t2 = static_cast<const array_data &>(*type);
    if (t2.size_is_infinite)
      throw array_type2t::inf_sized_array_excp();

    expr2tc arrsize = t2.array_size;
    simplify(arrsize);
    if (!is_constant_int2t(arrsize))
      throw array_type2t::dyn_sized_array_excp(arrsize);

    BigInt subsize = size_bits(t2.subtype);
    const constant_int2t &arrsize_int = to_constant_int2t(arrsize);
    return subsize * arrsize_int.value;
  }

  case type2t::struct_id:
  {
    // Compute the size of all members of this struct, and add padding bytes
    // so that they all start on wourd boundries. Also add any trailing bytes
    // necessary to make arrays align properly if malloc'd, see C89 6.3.3.4.
    BigInt accumulated_size = 0;
    for (auto const &it : to_struct_type(type).members)
      accumulated_size += size_bits(it);

    // At the end of that, the tests above should have rounded accumulated size
    // up to a size that contains the required trailing padding for array
    // allocation alignment.
    return accumulated_size == 0 ? 1 : accumulated_size;
  }

  case type2t::union_id:
  {
    // Very simple: the largest field size, rounded up to a word boundary for
    // array allocation alignment.
    BigInt max_size = 0;
    for (auto const &it : to_union_type(type).members)
      max_size = std::max(max_size, size_bits(it));
    return max_size;
  }

  default:
    log_error("Unrecognised type in size_typet::size_bits:\n{}", *type);
    abort();
  }
}

BigInt type_byte_size_bits(const type2tc &type, const namespacet *ns)
{
  return type_sizet(ns).size_bits(type);
}

BigInt type_byte_size(const type2tc &type, const namespacet *ns)
{
  BigInt bits = type_byte_size_bits(type, ns);

  return (bits + 7) / 8;
}

static expr2tc bitsize(BigInt n)
{
  return constant_int2tc(bitsize_type2(), std::move(n));
}

expr2tc type_sizet::size_bits_expr(const type2tc &type) const
{
  /* The structure of this function is the same as that of
   * size_bits(). We don't call that and just handle the exception
   * though, since that would mean we'd unnecessarily recurse multiple times
   * into the type. */

  switch (type->type_id)
  {
  // This is a gcc extension.
  // https://gcc.gnu.org/onlinedocs/gcc-4.8.0/gcc/Pointer-Arith.html
  case type2t::empty_id:
    return bitsize(1);

  case type2t::code_id:
    return bitsize(0);

  case type2t::symbol_id:
    if (ns)
      return size_bits_expr(ns->follow(type));
    log_error("Symbolic type id in type_size_t::size_bits_expr\n{}", *type);
    abort();

  case type2t::cpp_name_id:
    if (ns)
      return size_bits_expr(ns->follow(type));
    log_error("Symbolic C++ type id in type_size_t::size_bits_expr\n{}", *type);
    abort();

  case type2t::bool_id:
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
  case type2t::pointer_id:
    return bitsize(type->get_width());

  case type2t::array_id:
  case type2t::vector_id:
  {
    const array_data &t2 = static_cast<const array_data &>(*type);

    if (t2.size_is_infinite)
      throw array_type2t::inf_sized_array_excp();

    expr2tc arrsize = t2.array_size;
    simplify(arrsize);

    expr2tc subsize = size_bits_expr(t2.subtype);
    simplify(subsize);

    if (is_constant_int2t(arrsize) && is_constant_int2t(subsize))
      return bitsize(
        to_constant_int2t(subsize).value * to_constant_int2t(arrsize).value);

    type2tc t = bitsize_type2();
    if (arrsize->type != t)
      arrsize = typecast2tc(t, arrsize);
    return mul2tc(t, subsize, arrsize);
  }

  case type2t::struct_id:
  {
    // Compute the size of all members of this struct, and add padding bytes
    // so that they all start on wourd boundries. Also add any trailing bytes
    // necessary to make arrays align properly if malloc'd, see C89 6.3.3.4.
    BigInt acc_cnst = 0;
    expr2tc acc_dyn;
    type2tc t = bitsize_type2();
    for (const type2tc &member : to_struct_type(type).members)
    {
      expr2tc s = size_bits_expr(member);
      if (is_constant_int2t(s))
        acc_cnst += to_constant_int2t(s).value;
      else if (acc_dyn)
        acc_dyn = add2tc(t, acc_dyn, s);
      else
        acc_dyn = s;
    }

    // At the end of that, the tests above should have rounded accumulated size
    // up to a size that contains the required trailing padding for array
    // allocation alignment.

    if (!acc_dyn)
      return bitsize(acc_cnst);

    if (acc_cnst == 0)
      return acc_dyn;

    return add2tc(t, bitsize(acc_cnst), acc_dyn);
  }

  case type2t::union_id:
  {
    // Very simple: the largest field size, rounded up to a word boundary for
    // array allocation alignment.
    BigInt max_cnst = 0;
    expr2tc max_dyn;
    type2tc t = bitsize_type2();
    for (const type2tc &elem : to_union_type(type).members)
    {
      expr2tc s = size_bits_expr(elem);
      if (is_constant_int2t(s))
        max_cnst = std::max(max_cnst, to_constant_int2t(s).value);
      else if (max_dyn)
        max_dyn = if2tc(t, greaterthan2tc(max_dyn, s), max_dyn, s);
      else
        max_dyn = s;
    }

    if (!max_dyn)
      return bitsize(max_cnst);

    if (max_cnst == 0)
      return max_dyn;

    expr2tc c = bitsize(max_cnst);
    return if2tc(t, greaterthan2tc(max_dyn, c), max_dyn, c);
  }

  default:
    log_error("Unrecognised type in type_size_t::size_bits_expr:\n{}", *type);
    abort();
  }
}

expr2tc type_byte_size_bits_expr(const type2tc &type, const namespacet *ns)
{
  return type_sizet(ns).size_bits_expr(type);
}

expr2tc type_byte_size_expr(const type2tc &type, const namespacet *ns)
{
  expr2tc n = type_byte_size_bits_expr(type, ns);
  type2tc t = size_type2();
  if (is_constant_int2t(n))
    return gen_long(t, (to_constant_int2t(n).value + 7) / 8);
  type2tc s = bitsize_type2();
  return typecast2tc(t, div2tc(s, add2tc(s, n, bitsize(7)), bitsize(8)));
}

expr2tc type_sizet::pointer_offset_bits(const expr2tc &expr) const
{
  if (is_symbol2t(expr))
    return bitsize(0);

  if (is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);

    expr2tc sub_size;
    if (is_array_type(index.source_value))
    {
      const array_type2t &arr_type = to_array_type(index.source_value->type);
      sub_size = size_bits_expr(arr_type.subtype);
    }
    else
    {
      throw std::runtime_error(
        "Unexpected index type in computer_pointer_offset");
    }

    /* Expression needs to be of the bitwidth capable of addressing bits in the
     * standard address space. */
    expr2tc result;
    type2tc t = bitsize_type2();
    if (is_constant_int2t(sub_size) && is_constant_int2t(index.index))
    {
      const constant_int2t &index_val = to_constant_int2t(index.index);
      const constant_int2t &ss_val = to_constant_int2t(sub_size);
      result = bitsize(ss_val.value * index_val.value);
    }
    else
    {
      /* Non constant, create multiply. */
      expr2tc the_index = index.index;
      if (sub_size->type != t)
        sub_size = typecast2tc(t, sub_size);
      if (the_index->type != t)
        the_index = typecast2tc(t, the_index);

      result = mul2tc(t, sub_size, the_index);
    }

    // Also accumulate any pointer offset in the source object.
    result = add2tc(t, result, pointer_offset_bits(index.source_value));

    return result;
  }

  if (is_member2t(expr))
  {
    const member2t &memb = to_member2t(expr);

    BigInt result;
    if (is_struct_type(memb.source_value->type))
    {
      result = member_offset_bits(memb.source_value->type, memb.member);
    }
    else
    {
      result = 0; // Union offsets are always 0.
    }

    // Also accumulate any pointer offset in the source object.
    expr2tc res_expr = bitsize(result);
    res_expr =
      add2tc(res_expr->type, res_expr, pointer_offset_bits(memb.source_value));

    return res_expr;
  }

  if (is_constant_expr(expr))
  {
    // This is a constant struct, array, union, string, etc. There's nothing
    // at a lower level; the offset is zero.
    return bitsize(0);
  }

  if (is_typecast2t(expr))
  {
    // Blast straight through.
    return pointer_offset_bits(to_typecast2t(expr).from);
  }

  if (is_dynamic_object2t(expr))
  {
    // This is a dynamic object represented something allocated; from the static
    // pointer analysis. Assume that this is the bottom of the expression.
    return bitsize(0);
  }

  if (is_dereference2t(expr))
  {
    // This is a dereference at the base of a set of index/members. Here, we
    // can in theory end up evaluating across a large set of object types. So
    // there's no point continuing further or attempting to dereference, leave
    // it up to the caller to handle that.
    return bitsize(0);
  }

  log_error("compute_pointer_offset, unexpected irep:\n{}", expr->pretty());
  abort();
}

expr2tc compute_pointer_offset_bits(const expr2tc &expr, const namespacet *ns)
{
  return type_sizet(ns).pointer_offset_bits(expr);
}

expr2tc compute_pointer_offset(const expr2tc &expr, const namespacet *ns)
{
  expr2tc bits = compute_pointer_offset_bits(expr, ns);
  expr2tc bytes = div2tc(bits->type, bits, bitsize(8));
  return typecast2tc(size_type2(), bytes);
}

const expr2tc &get_base_object(const expr2tc &expr)
{
  if (is_index2t(expr))
    return get_base_object(to_index2t(expr).source_value);

  if (is_member2t(expr))
    return get_base_object(to_member2t(expr).source_value);

  if (is_typecast2t(expr))
    return get_base_object(to_typecast2t(expr).from);

  if (is_address_of2t(expr))
    return get_base_object(to_address_of2t(expr).ptr_obj);

  if (is_dereference2t(expr))
    return get_base_object(to_dereference2t(expr).value);

  return expr;
}
