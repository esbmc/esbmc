/*******************************************************************\

   Module: Pointer Logic

   Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr.h>
#include <irep2/irep2_utils.h>
#include <util/std_types.h>
#include <util/type_byte_size.h>
#include <util/message/format.h>

BigInt member_offset(const type2tc &type, const irep_idt &member)
{
  return member_offset_bits(type, member) / 8;
}

BigInt member_offset_bits(const type2tc &type, const irep_idt &member)
{
  BigInt result = 0;

  // empty union generate an array
  if(!is_struct_type(type))
    return result;

  unsigned idx = 0;
  const struct_type2t &thetype = to_struct_type(type);
  for(auto const &it : thetype.members)
  {
    if(thetype.member_names[idx] == member.as_string())
      break;

    result += type_byte_size_bits(it);
    idx++;
  }

  return result;
}

BigInt type_byte_size_default(const type2tc &type, const BigInt &defaultval)
{
  try
  {
    return type_byte_size(type);
  }
  catch(const array_type2t::dyn_sized_array_excp &e)
  {
    return defaultval;
  }
}

BigInt type_byte_size(const type2tc &type)
{
  BigInt bits = type_byte_size_bits(type);

  return (bits + 7) / 8;
}

BigInt type_byte_size_bits(const type2tc &type)
{
  switch(type->type_id)
  {
  // This is a gcc extension.
  // https://gcc.gnu.org/onlinedocs/gcc-4.8.0/gcc/Pointer-Arith.html
  case type2t::empty_id:
    return 1;

  case type2t::code_id:
    return 0;

  case type2t::symbol_id:
    assert(
      0 &&
      fmt::format("Symbolic type id in type_byte_size\n{}", *type).c_str());

  case type2t::cpp_name_id:
    assert(
      0 &&
      fmt::format("C++ symbolic type id in type_byte_size\n{}", *type).c_str());

  case type2t::bool_id:
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
  case type2t::pointer_id:
    return type->get_width();

  case type2t::string_id:
    // TODO: Strings of wchar will return the wrong result here
    return to_string_type(type).width * config.ansi_c.char_width;

  case type2t::vector_id:
  {
    // Attempt to compute constant array offset. If we can't, we can't
    // reasonably return anything anyway, so throw.
    const vector_type2t &t2 = to_vector_type(type);
    if(t2.size_is_infinite)
      throw new array_type2t::inf_sized_array_excp();

    expr2tc arrsize = t2.array_size;
    simplify(arrsize);
    if(!is_constant_int2t(arrsize))
      throw new array_type2t::dyn_sized_array_excp(arrsize);

    BigInt subsize = type_byte_size_bits(t2.subtype);
    const constant_int2t &arrsize_int = to_constant_int2t(arrsize);
    return subsize * arrsize_int.value;
  }

  case type2t::array_id:
  {
    // Attempt to compute constant array offset. If we can't, we can't
    // reasonably return anything anyway, so throw.
    const array_type2t &t2 = to_array_type(type);
    if(t2.size_is_infinite)
      throw array_type2t::inf_sized_array_excp();

    expr2tc arrsize = t2.array_size;
    simplify(arrsize);
    if(!is_constant_int2t(arrsize))
      throw array_type2t::dyn_sized_array_excp(arrsize);

    BigInt subsize = type_byte_size_bits(t2.subtype);
    const constant_int2t &arrsize_int = to_constant_int2t(arrsize);
    return subsize * arrsize_int.value;
  }

  case type2t::struct_id:
  {
    // Compute the size of all members of this struct, and add padding bytes
    // so that they all start on wourd boundries. Also add any trailing bytes
    // necessary to make arrays align properly if malloc'd, see C89 6.3.3.4.
    BigInt accumulated_size = 0;
    for(auto const &it : to_struct_type(type).members)
      accumulated_size += type_byte_size_bits(it);

    // At the end of that, the tests above should have rounded accumulated size
    // up to a size that contains the required trailing padding for array
    // allocation alignment.
    return accumulated_size;
  }

  case type2t::union_id:
  {
    // Very simple: the largest field size, rounded up to a word boundry for
    // array allocation alignment.
    BigInt max_size = 0;
    for(auto const &it : to_union_type(type).members)
      max_size = std::max(max_size, type_byte_size_bits(it));
    return max_size;
  }

  default:
    assert(
      0 && fmt::format("Unrecognised type in type_byte_size_bits:\n{}", *type)
             .c_str());
    abort();
  }
}

expr2tc compute_pointer_offset_bits(const expr2tc &expr)
{
  if(is_symbol2t(expr))
    return gen_ulong(0);

  if(is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);

    BigInt sub_size;
    if(is_array_type(index.source_value))
    {
      const array_type2t &arr_type = to_array_type(index.source_value->type);
      sub_size = type_byte_size_bits(arr_type.subtype);
    }
    else if(is_string_type(index.source_value))
    {
      sub_size = 64;
    }
    else
    {
      throw std::runtime_error(
        "Unexpected index type in computer_pointer_offset");
    }

    expr2tc result;
    if(is_constant_int2t(index.index))
    {
      const constant_int2t &index_val = to_constant_int2t(index.index);
      result = gen_ulong(BigInt(sub_size * index_val.value).to_uint64());
    }
    else
    {
      // Non constant, create multiply.
      // Index operand needs to be the bitwidth of a 'long'.
      expr2tc zero_ulong = gen_ulong(0);
      expr2tc the_index = index.index;
      if(the_index->type != zero_ulong->type)
        the_index = typecast2tc(zero_ulong->type, the_index);

      constant_int2tc tmp_size(zero_ulong->type, sub_size);
      result = mul2tc(zero_ulong->type, tmp_size, the_index);
    }

    // Also accumulate any pointer offset in the source object.
    result = add2tc(
      result->type, result, compute_pointer_offset_bits(index.source_value));

    return result;
  }

  if(is_member2t(expr))
  {
    const member2t &memb = to_member2t(expr);

    BigInt result;
    if(is_struct_type(memb.source_value->type))
    {
      result = member_offset_bits(memb.source_value->type, memb.member);
    }
    else
    {
      result = 0; // Union offsets are always 0.
    }

    // Also accumulate any pointer offset in the source object.
    expr2tc res_expr = gen_ulong(result.to_uint64());
    res_expr = add2tc(
      res_expr->type, res_expr, compute_pointer_offset_bits(memb.source_value));

    return res_expr;
  }

  if(is_constant_expr(expr))
  {
    // This is a constant struct, array, union, string, etc. There's nothing
    // at a lower level; the offset is zero.
    return gen_ulong(0);
  }

  if(is_typecast2t(expr))
  {
    // Blast straight through.
    return compute_pointer_offset_bits(to_typecast2t(expr).from);
  }

  if(is_dynamic_object2t(expr))
  {
    // This is a dynamic object represented something allocated; from the static
    // pointer analysis. Assume that this is the bottom of the expression.
    return gen_ulong(0);
  }

  if(is_dereference2t(expr))
  {
    // This is a dereference at the base of a set of index/members. Here, we
    // can in theory end up evaluating across a large set of object types. So
    // there's no point continuing further or attempting to dereference, leave
    // it up to the caller to handle that.
    return gen_ulong(0);
  }

  throw std::runtime_error(fmt::format(
    "compute_pointer_offset, unexpected irep:\n{}", expr->pretty()));
}

expr2tc compute_pointer_offset(const expr2tc &expr)
{
  expr2tc pointer_offset_bits = compute_pointer_offset_bits(expr);

  expr2tc result =
    div2tc(pointer_offset_bits->type, pointer_offset_bits, gen_ulong(8));
  return result;
}

const expr2tc &get_base_object(const expr2tc &expr)
{
  if(is_index2t(expr))
    return get_base_object(to_index2t(expr).source_value);

  if(is_member2t(expr))
    return get_base_object(to_member2t(expr).source_value);

  if(is_typecast2t(expr))
    return get_base_object(to_typecast2t(expr).from);

  if(is_address_of2t(expr))
    return get_base_object(to_address_of2t(expr).ptr_obj);

  return expr;
}

const irep_idt get_string_argument(const expr2tc &expr)
{
  // Remove typecast
  if(is_typecast2t(expr))
    return get_string_argument(to_typecast2t(expr).from);

  // Remove address_of
  if(is_address_of2t(expr))
    return get_string_argument(to_address_of2t(expr).ptr_obj);

  // Remove index
  if(is_index2t(expr))
    return get_string_argument(to_index2t(expr).source_value);

  if(is_constant_string2t(expr))
    return to_constant_string2t(expr).value;

  return "";
}
