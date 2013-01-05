/*******************************************************************\

Module: Pointer Logic

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <expr.h>
#include <arith_tools.h>
#include <std_types.h>
#include <ansi-c/c_types.h>

#include "pointer_offset_size.h"

mp_integer member_offset(
  const struct_type2t &type,
  const irep_idt &member)
{
  mp_integer result=0;
  unsigned bit_field_bits=0, idx = 0;

  forall_types(it, type.members) {
    if (type.member_names[idx] == member.as_string())
      break;

    // XXXjmorse - just assume we break horribly on bitfields.
#if 0
    if(it->get_bool("#is_bit_field"))
    {
      bit_field_bits+=binary2integer(it->type().get("width").as_string(), 2).to_long();
    }
#endif

    mp_integer sub_size=pointer_offset_size(**it);
    if (sub_size==-1)
      return -1; // give up

    result += sub_size;
    idx++;
  }

  return result;
}

mp_integer pointer_offset_size(const type2t &type)
{

  return type.get_width() / 8;
}

expr2tc
compute_pointer_offset(const expr2tc &expr)
{
  if (is_symbol2t(expr))
    return zero_uint;
  else if (is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);
    mp_integer sub_size;
    if (is_array_type(index.source_value)) {
      const array_type2t &arr_type = to_array_type(index.source_value->type);
      sub_size = pointer_offset_size(*arr_type.subtype.get());
    } else if (is_string_type(index.source_value)) {
      sub_size = 8;
    } else {
      std::cerr << "Unexpected index type in computer_pointer_offset";
      std::cerr << std::endl;
      abort();
    }

    expr2tc result;
    if (is_constant_int2t(index.index)) {
      const constant_int2t &index_val = to_constant_int2t(index.index);
      result = constant_int2tc(get_uint_type(32),
                               sub_size * index_val.constant_value);
    } else {
      // Non constant, create multiply.
      constant_int2tc tmp_size(uint_type2(), sub_size);
      result = mul2tc(uint_type2(), tmp_size, index.index);
    }

    return result;
  }
  else if (is_member2t(expr))
  {
    const member2t &memb = to_member2t(expr);

    mp_integer result;
    if (is_struct_type(expr)) {
      const struct_type2t &type = to_struct_type(expr->type);
      result = member_offset(type, memb.member);
    } else {
      result = 0; // Union offsets are always 0.
    }

    return constant_int2tc(uint_type2(), result);
  }
  else if (is_constant_array2t(expr))
  {
    // Some code, somewhere, is renaming a constant array into a dereference
    // target. The offset into the base object is nothing.
    return zero_uint;
  }
  else
  {
    std::cerr << "compute_pointer_offset, unexpected irep:" << std::endl;
    std::cerr << expr->pretty() << std::endl;
    abort();
  }
}

const expr2tc &
get_base_object(const expr2tc &expr)
{

  if (is_index2t(expr)) {
    return get_base_object(to_index2t(expr).source_value);
  } else if (is_member2t(expr)) {
    return get_base_object(to_member2t(expr).source_value);
  } else {
    return expr;
  }
}
