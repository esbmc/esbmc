/*******************************************************************\

   Module: Pointer Logic

   Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr.h>
#include <util/irep2_utils.h>
#include <util/std_types.h>
#include <util/type_byte_size.h>

static inline void
round_up_to_word(mp_integer &mp)
{
  const unsigned int word_bytes = config.ansi_c.word_size / 8;
  const unsigned int align_mask = word_bytes - 1;

  if (mp == 0) {
    return;
  } else if (mp < word_bytes) {
    mp = mp_integer(word_bytes);
  // Or if it's an array of chars etc. that doesn't end on a boundry,
  } else if (mp.to_ulong() & align_mask) {
    mp += word_bytes - (mp.to_ulong() & align_mask);
  }
}

static inline void
round_up_to_int64(mp_integer &mp)
{
  const unsigned int word_bytes = 8;
  const unsigned int align_mask = 7;

  if (mp == 0) {
    return;
  } else if (mp < word_bytes) {
    mp = mp_integer(word_bytes);
  // Or if it's an array of chars etc. that doesn't end on a boundry,
  } else if (mp.to_ulong() & align_mask) {
    mp += word_bytes - (mp.to_ulong() & align_mask);
  }
}

mp_integer
member_offset(const type2tc &type, const irep_idt &member)
{
  mp_integer result = 0;
  unsigned idx = 0;

  const struct_type2t &thetype = to_struct_type(type);

  for(auto const &it : thetype.members)
  {
    // If the current field is 64 bits, and we're on a 32 bit machine, then we
    // _must_ round up to 64 bits now.
    if (is_scalar_type(it) && !is_code_type(it) &&
        (it)->get_width() > 32 && config.ansi_c.word_size == 32)
      round_up_to_int64(result);

    if (is_structure_type(it))
      round_up_to_int64(result);

    if (is_array_type(it) && to_array_type(it).subtype->get_width() > 32)
      round_up_to_int64(result);

    if (thetype.member_names[idx] == member.as_string())
      break;

    // XXX 100% unhandled: bitfields.

    mp_integer sub_size = type_byte_size(it);
    // Handle padding: we need to observe the usual struct constraints.
    round_up_to_word(sub_size);

    result += sub_size;
    idx++;
  }

  assert(idx != thetype.members.size() && "Attempted to find member offset of "
         "member not in a struct");

  return result;
}

mp_integer
type_byte_size_default(const type2tc &type, const mp_integer& defaultval)
{
  try {
    return type_byte_size(type);
  } catch (array_type2t::dyn_sized_array_excp *e) {
    return defaultval;
  }
}

mp_integer
type_byte_size(const type2tc &type)
{

  switch (type.get()->type_id) {
  case type2t::bool_id:
    return 1;
  case type2t::empty_id:
    std::cerr << "Void type id in type_byte_size" <<std::endl;
    abort();
  case type2t::symbol_id:
    std::cerr << "Symbolic type id in type_byte_size" <<std::endl;
    type.get()->dump();
    abort();
  case type2t::code_id:
    // In C++, methods are struct fields.
    return 0;
    abort();
  case type2t::cpp_name_id:
    std::cerr << "C++ symbolic type id in type_byte_size" <<std::endl;
    type.get()->dump();
    abort();
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
    return mp_integer(type.get()->get_width() / 8);
  case type2t::pointer_id:
    return mp_integer(config.ansi_c.pointer_width / 8);
  case type2t::string_id:
  {
    const string_type2t &t2 = to_string_type(type);
    return mp_integer(t2.width);
  }
  case type2t::array_id:
  {
    // Array width is the subtype width, rounded up to whatever alignment is
    // necessary, multiplied by the size.

    // type_byte_size will handle all alignment and trailing padding byte
    // problems.
    const array_type2t &t2 = to_array_type(type);
    mp_integer subsize = type_byte_size(t2.subtype);

    // Attempt to compute constant array offset. If we can't, we can't
    // reasonably return anything anyway, so throw.
    expr2tc arrsize = t2.array_size;
    if (!t2.size_is_infinite)
    {
      simplify(arrsize);

      if (!is_constant_int2t(arrsize))
        throw new array_type2t::dyn_sized_array_excp(arrsize);
    } else {
      throw new array_type2t::inf_sized_array_excp();
    }

    const constant_int2t &arrsize_int = to_constant_int2t(arrsize);
    return subsize * arrsize_int.value;
  }
  case type2t::struct_id:
  {
    // Compute the size of all members of this struct, and add padding bytes
    // so that they all start on wourd boundries. Also add any trailing bytes
    // necessary to make arrays align properly if malloc'd, see C89 6.3.3.4.

    const struct_type2t &t2 = to_struct_type(type);
    mp_integer accumulated_size(0);
    for(auto const &it : t2.members)
    {
      // If the current field is 64 bits, and we're on a 32 bit machine, then we
      // _must_ round up to 64 bits now. Also guard against symbolic types
      // as operands.
      if (is_scalar_type(it) && !is_code_type(it) &&
          (it)->get_width() > 32 && config.ansi_c.word_size == 32)
        round_up_to_int64(accumulated_size);

      // While we're at it, round any struct/union up to 64 bit alignment too,
      // as that might require such alignment due to internal doubles.
      if (is_structure_type(it))
        round_up_to_int64(accumulated_size);

      // Also arrays of int64's. One onders why I bother.
      if (is_array_type(it) && to_array_type(*it).subtype->get_width() > 32)
        round_up_to_int64(accumulated_size);

      mp_integer memb_size = type_byte_size(it);

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
    const union_type2t &t2 = to_union_type(type);
    mp_integer max_size(0);
    for(auto const &it : t2.members)
    {
      mp_integer memb_size = type_byte_size(it);
      max_size = std::max(max_size, memb_size);
    }

    round_up_to_word(max_size);
    return max_size;
  }
  default:
    std::cerr << "Unrecognised type in type_byte_size:" << std::endl;
    type.get()->dump();
    abort();
  }
}

expr2tc
compute_pointer_offset(const expr2tc &expr)
{
  if (is_symbol2t(expr))
    return gen_ulong(0);
  else if (is_index2t(expr)) {
    const index2t &index = to_index2t(expr);
    mp_integer sub_size;
    if (is_array_type(index.source_value)) {
      const array_type2t &arr_type = to_array_type(index.source_value->type);
      sub_size = type_byte_size(arr_type.subtype);
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
      result =
        gen_ulong(BigInt(sub_size * index_val.value).to_ulong());
    } else {
      // Non constant, create multiply.
      // Index operand needs to be the bitwidth of a 'long'.
      expr2tc zero_ulong = gen_ulong(0);
      expr2tc the_index = index.index;
      if (the_index->type != zero_ulong->type)
        the_index = typecast2tc(zero_ulong->type, the_index);

      constant_int2tc tmp_size(zero_ulong->type, sub_size);
      result = mul2tc(zero_ulong->type, tmp_size, the_index);
    }

    // Also accumulate any pointer offset in the source object.
    result = add2tc(result->type, result,
                    compute_pointer_offset(index.source_value));

    return result;
  } else if (is_member2t(expr))   {
    const member2t &memb = to_member2t(expr);

    mp_integer result;
    if (is_struct_type(memb.source_value->type)) {
      result = member_offset(memb.source_value->type, memb.member);
    } else {
      result = 0; // Union offsets are always 0.
    }

    // Also accumulate any pointer offset in the source object.
    expr2tc res_expr = gen_ulong(result.to_ulong());
    res_expr = add2tc(res_expr->type, res_expr,
                      compute_pointer_offset(memb.source_value));

    return res_expr;
  } else if (is_constant_expr(expr)) {
    // This is a constant struct, array, union, string, etc. There's nothing
    // at a lower level; the offset is zero.
    return gen_ulong(0);
  } else if (is_typecast2t(expr)) {
    // Blast straight through.
    return compute_pointer_offset(to_typecast2t(expr).from);
  } else if (is_dynamic_object2t(expr)) {
    // This is a dynamic object represented something allocated; from the static
    // pointer analysis. Assume that this is thet bottom of the expression.
    return gen_ulong(0);
  } else if (is_dereference2t(expr)) {
    // This is a dereference at the base of a set of index/members. Here, we
    // can in theory end up evaluating across a large set of object types. So
    // there's no point continuing further or attempting to dereference, leave
    // it up to the caller to handle that.
    return gen_ulong(0);
  } else {
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
