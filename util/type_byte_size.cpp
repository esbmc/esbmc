/*******************************************************************\

   Module: Pointer Logic

   Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <expr.h>
#include <arith_tools.h>
#include <std_types.h>
#include <ansi-c/c_types.h>

#include "type_byte_size.h"

static inline void
round_up_to_word(mp_integer &mp)
{
  const unsigned int word_bytes = config.ansi_c.word_size / 8;
  const unsigned int align_mask = word_bytes - 1;

  if (mp < word_bytes) {
    mp = mp_integer(word_bytes);
  // Or if it's an array of chars etc. that doesn't end on a boundry,
  } else if (mp.to_ulong() & align_mask) {
    mp += word_bytes - (mp.to_ulong() & align_mask);
  }

  return;
}

mp_integer
member_offset(const struct_type2t &type, const irep_idt &member)
{
  mp_integer result = 0;
  unsigned bit_field_bits = 0, idx = 0;

  forall_types(it, type.members) {
    if (type.member_names[idx] == member.as_string())
      break;

    // XXX 100% unhandled: bitfields.

    mp_integer sub_size = type_byte_size(**it);
    // Handle padding: we need to observe the usual struct constraints.
    round_up_to_word(sub_size);

    result += sub_size;
    idx++;
  }

  assert(idx != type.members.size() && "Attempted to find member offset of "
         "member not in a struct");

  return result;
}

mp_integer
type_byte_size(const type2t &type)
{

  switch (type.type_id) {
  case type2t::bool_id:
    return 1;
  case type2t::empty_id:
    std::cerr << "Void type id in type_byte_size" <<std::endl;
    abort();
  case type2t::symbol_id:
    std::cerr << "Symbolic type id in type_byte_size" <<std::endl;
    type.dump();
    abort();
  case type2t::code_id:
    std::cerr << "Code type id in type_byte_size" <<std::endl;
    type.dump();
    abort();
  case type2t::cpp_name_id:
    std::cerr << "C++ symbolic type id in type_byte_size" <<std::endl;
    type.dump();
    abort();
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
    return mp_integer(type.get_width() / 8);
  case type2t::pointer_id:
    return mp_integer(config.ansi_c.pointer_width / 8);
  case type2t::string_id:
  {
    const string_type2t &t2 = static_cast<const string_type2t&>(type);
    return mp_integer(t2.width);
  }
  case type2t::array_id:
  {
    // Array width is the subtype width, rounded up to whatever alignment is
    // necessary, multiplied by the size.

    // type_byte_size will handle all alignment and trailing padding byte
    // problems.
    const array_type2t &t2 = static_cast<const array_type2t&>(type);
    mp_integer subsize = type_byte_size(*t2.subtype);

    // Attempt to compute constant array offset. If we can't, we can't
    // reasonably return anything anyway, so throw.
    expr2tc arrsize;
    if (!t2.size_is_infinite) {
     arrsize = t2.array_size->simplify();
      if (is_nil_expr(arrsize))
        arrsize = t2.array_size;

      if (!is_constant_int2t(arrsize))
        throw new array_type2t::dyn_sized_array_excp(arrsize);
    } else {
      throw new array_type2t::inf_sized_array_excp();
    }

    const constant_int2t &arrsize_int = to_constant_int2t(arrsize);
    return subsize * arrsize_int.constant_value;
  }
  case type2t::struct_id:
  {
    // Compute the size of all members of this struct, and add padding bytes
    // so that they all start on wourd boundries. Also add any trailing bytes
    // necessary to make arrays align properly if malloc'd, see C89 6.3.3.4.

    const struct_type2t &t2 = static_cast<const struct_type2t&>(type);
    mp_integer accumulated_size(0);
    forall_types(it, t2.members) {
      mp_integer memb_size = type_byte_size(**it);

      round_up_to_word(memb_size);

      accumulated_size += memb_size;
    }

    // At the end of that, the tests above should have rounded accumulated size
    // up to a size that contains the required trailing padding for array
    // allocation alignment.
    assert((accumulated_size % (config.ansi_c.word_size / 8)) == 0);
    return accumulated_size;
  }
  case type2t::union_id:
  {
    // Very simple: the largest field size, rounded up to a word boundry for
    // array allocation alignment.
    const union_type2t &t2 = static_cast<const union_type2t&>(type);
    mp_integer max_size(0);
    forall_types(it, t2.members) {
      mp_integer memb_size = type_byte_size(**it);
      max_size = std::max(max_size, memb_size);
    }

    round_up_to_word(max_size);
    return max_size;
  }
  default:
    std::cerr << "Unrecognised type in type_byte_size:" << std::endl;
    type.dump();
    abort();
  }
}

expr2tc
compute_pointer_offset(const expr2tc &expr)
{
  if (is_symbol2t(expr))
    return zero_uint;
  else if (is_index2t(expr)) {
    const index2t &index = to_index2t(expr);
    mp_integer sub_size;
    if (is_array_type(index.source_value)) {
      const array_type2t &arr_type = to_array_type(index.source_value->type);
      sub_size = type_byte_size(*arr_type.subtype.get());
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
  } else if (is_member2t(expr))   {
    const member2t &memb = to_member2t(expr);

    mp_integer result;
    if (is_struct_type(expr)) {
      const struct_type2t &type = to_struct_type(expr->type);
      result = member_offset(type, memb.member);
    } else {
      result = 0; // Union offsets are always 0.
    }

    return constant_int2tc(uint_type2(), result);
  } else if (is_constant_array2t(expr))   {
    // Some code, somewhere, is renaming a constant array into a dereference
    // target. The offset into the base object is nothing.
    return zero_uint;
  } else   {
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
  } else if (is_typecast2t(expr)) {
    return get_base_object(to_typecast2t(expr).from);
  } else {
    return expr;
  }
}
