#include <solvers/smt/smt_conv.h>
#include <util/type_byte_size.h>

smt_astt smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);
  expr2tc source = data.source_value;
  expr2tc offs = data.source_offset;

  assert(!is_array_type(source));

  unsigned int src_width = source->type->get_width();

  if (int_encoding)
    return convert_byte_extract_int_mode(data, source, offs, src_width);
  else
    return convert_byte_extract_bv_mode(data, source, offs, src_width);
}

smt_astt smt_convt::convert_byte_extract_int_mode(
  const byte_extract2t &data,
  expr2tc source,
  expr2tc offs,
  unsigned int src_width)
{
  // Convert source to integer representation if needed
  if (!is_number_type(source->type))
  {
    // For non-numeric types, we need to interpret them as mathematical integers
    source = typecast2tc(get_uint_type(src_width), source);
  }

  if (!is_constant_int2t(offs))
  {
    // Non-constant offset case - use mathematical operations
    // to simulate bit shifting and extraction

    // Ensure offset is the same type as source for arithmetic operations
    if (offs->type->get_width() != source->type->get_width())
      offs = typecast2tc(source->type, offs);

    // Handle endianness by adjusting the offset
    if (data.big_endian)
    {
      auto data_size = type_byte_size(source->type);
      expr2tc data_size_expr = constant_int2tc(source->type, data_size - 1);
      offs = sub2tc(source->type, data_size_expr, offs);
    }

    // Convert byte offset to bit offset (multiply by bits per byte)
    expr2tc bit_offset = mul2tc(
      offs->type,
      offs,
      constant_int2tc(offs->type, BigInt(config.ansi_c.char_width)));

    // Simulate right shift using division by 2^bit_offset
    expr2tc shifted_source = create_int_right_shift(source, bit_offset);

    // Extract bottom byte using bitwise AND with byte mask (2^char_width - 1)
    BigInt byte_mask_value =
      (BigInt(1) << config.ansi_c.char_width) - BigInt(1);
    expr2tc byte_mask = constant_int2tc(source->type, byte_mask_value);
    expr2tc extracted_byte = bitand2tc(source->type, shifted_source, byte_mask);

    return convert_ast(extracted_byte);
  }
  else
  {
    // Constant offset case - can use direct mathematical operations
    const constant_int2t &intref = to_constant_int2t(offs);
    unsigned int byte_offset = intref.value.to_uint64();

    // Calculate the divisor for extracting the specific byte
    BigInt divisor = BigInt(1);
    unsigned int shift_amount;

    if (!data.big_endian)
    {
      // Little endian: byte 0 is least significant
      shift_amount = byte_offset * config.ansi_c.char_width;
    }
    else
    {
      // Big endian: byte 0 is most significant
      unsigned int total_bytes = src_width / config.ansi_c.char_width;
      if (byte_offset >= total_bytes)
      {
        // Out of bounds access
        smt_sortt s = mk_int_sort();
        return mk_smt_symbol("out_of_bounds_byte_extract", s);
      }
      shift_amount = (total_bytes - 1 - byte_offset) * config.ansi_c.char_width;
    }

    // Calculate 2^shift_amount
    for (unsigned int i = 0; i < shift_amount; i++)
      divisor = divisor * BigInt(2);

    // Perform integer division to simulate right shift
    if (shift_amount > 0)
    {
      expr2tc divisor_expr = constant_int2tc(source->type, divisor);
      source = div2tc(source->type, source, divisor_expr);
    }

    // Extract bottom byte using bitwise AND with byte mask (2^char_width - 1)
    BigInt byte_mask_value =
      (BigInt(1) << config.ansi_c.char_width) - BigInt(1);
    expr2tc byte_mask = constant_int2tc(source->type, byte_mask_value);
    expr2tc result = bitand2tc(source->type, source, byte_mask);

    return convert_ast(result);
  }
}

smt_astt smt_convt::convert_byte_extract_bv_mode(
  const byte_extract2t &data,
  expr2tc source,
  expr2tc offs,
  unsigned int src_width)
{
  if (!is_bv_type(source->type) && !is_fixedbv_type(source->type))
    source = bitcast2tc(get_uint_type(src_width), source);

  if (!is_constant_int2t(offs))
  {
    // The approach: the argument is now a bitvector. Just shift it the
    // appropriate amount, according to the source offset, and select out the
    // bottom byte.
    if (offs->type->get_width() != src_width)
      offs = typecast2tc(source->type, data.source_offset);

    // Endian-ness: if we're in non-"native" endian-ness mode, then flip the
    // offset distance. The rest of these calculations will still apply.
    if (data.big_endian)
    {
      auto data_size = type_byte_size(source->type);
      expr2tc data_size_expr = constant_int2tc(source->type, data_size - 1);
      expr2tc sub = sub2tc(source->type, data_size_expr, offs);
      offs = sub;
    }

    offs = mul2tc(offs->type, offs, constant_int2tc(offs->type, BigInt(8)));

    expr2tc shr = lshr2tc(source->type, source, offs);
    smt_astt ext = convert_ast(shr);
    smt_astt res = mk_extract(ext, 7, 0);
    return res;
  }

  const constant_int2t &intref = to_constant_int2t(offs);

  unsigned width;
  width = data.source_value->type->get_width();

  unsigned int upper, lower;
  if (!data.big_endian)
  {
    upper = ((intref.value.to_uint64() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.value.to_uint64() * 8;             //i*w;
  }
  else
  {
    unsigned int max = width - 1;
    upper = max - (intref.value.to_uint64() * 8);           //max-(i*w);
    lower = max - ((intref.value.to_uint64() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  smt_astt source_ast = convert_ast(source);

  if (width <= upper)
  {
    smt_sortt s = mk_int_bv_sort(8);
    return mk_smt_symbol("out_of_bounds_byte_extract", s);
  }

  return mk_extract(source_ast, upper, lower);
}

expr2tc smt_convt::create_int_right_shift(expr2tc source, expr2tc shift_amount)
{
  // For non-constant shift amounts, we use conditional expressions
  // for common shift amounts (bit-aligned shifts from 0 to 64)

  expr2tc result = source;

  // Create conditional chain for shift amounts up to pointer width (architecture-dependent)
  for (size_t i = config.ansi_c.char_width; i <= config.ansi_c.pointer_width();
       i += config.ansi_c.char_width) // Only byte-aligned shifts for efficiency
  {
    expr2tc i_expr = constant_int2tc(shift_amount->type, BigInt(i));
    expr2tc condition = equality2tc(shift_amount, i_expr);

    BigInt divisor = BigInt(1);
    for (size_t j = 0; j < i; j++)
    {
      divisor = divisor * BigInt(2);
    }

    expr2tc divisor_expr = constant_int2tc(source->type, divisor);
    expr2tc shifted = div2tc(source->type, source, divisor_expr);

    result = if2tc(source->type, condition, shifted, result);
  }

  return result;
}

smt_astt smt_convt::convert_byte_update(const expr2tc &expr)
{
  if (int_encoding)
  {
    log_error("Can't byte update in integer mode; rerun in bitvector mode");
    abort();
  }

  const byte_update2t &data = to_byte_update2t(expr);
  assert(data.type == data.source_value->type);

  if (is_array_type(data.type))
  {
    type2tc subtype = to_array_type(data.type).subtype;
    BigInt sub_size_int = type_byte_size(subtype);
    assert(
      !sub_size_int.is_zero() &&
      "Unimplemented byte_update on array of zero-width elements");
    expr2tc sub_size = constant_int2tc(data.source_offset->type, sub_size_int);
    expr2tc index = div2tc(sub_size->type, data.source_offset, sub_size);
    expr2tc new_offs = modulus2tc(sub_size->type, data.source_offset, sub_size);
    expr2tc new_bu = byte_update2tc(
      subtype,
      index2tc(subtype, data.source_value, index),
      new_offs,
      data.update_value,
      data.big_endian);
    expr2tc with = with2tc(data.type, data.source_value, index, new_bu);
    return convert_ast(with);
  }

  if (!is_bv_type(data.type) && !is_fixedbv_type(data.type))
  {
    // This is a pointer or a bool, or something. We don't want to handle
    // casting of it in the body of this function, so wrap it up as a bitvector
    // and re-apply.
    type2tc bit_type = get_uint_type(data.type->get_width());
    expr2tc src_obj = bitcast2tc(bit_type, data.source_value);
    expr2tc new_update = byte_update2tc(
      bit_type,
      src_obj,
      data.source_offset,
      data.update_value,
      data.big_endian);
    expr2tc cast_back = bitcast2tc(data.type, new_update);
    return convert_ast(cast_back);
  }

  if (!is_constant_int2t(data.source_offset))
  {
    expr2tc source = data.source_value;
    unsigned int src_width = source->type->get_width();
    type2tc org_type;
    if (!is_unsignedbv_type(source))
    {
      org_type = source->type;
      source = bitcast2tc(get_uint_type(src_width), source);
    }

    expr2tc offs = data.source_offset;
    if (!is_unsignedbv_type(offs) || offs->type->get_width() != src_width)
      offs = typecast2tc(get_uint_type(src_width), offs);

    // Endian-ness: if we're in non-"native" endian-ness mode, then flip the
    // offset distance. The rest of these calculations will still apply.
    if (data.big_endian)
    {
      auto data_size = type_byte_size(source->type);
      expr2tc data_size_expr = constant_int2tc(source->type, data_size - 1);
      expr2tc sub = sub2tc(source->type, data_size_expr, offs);
      offs = sub;
    }

    expr2tc update = data.update_value;
    if (!is_unsignedbv_type(update) || update->type->get_width() != src_width)
      update = typecast2tc(
        get_uint_type(src_width),
        bitcast2tc(get_uint_type(update->type->get_width()), update));

    // The approach: mask, shift and or. Quite inefficient.

    expr2tc eight = constant_int2tc(get_uint_type(src_width), BigInt(8));
    expr2tc effs = constant_int2tc(eight->type, BigInt(255));
    offs = mul2tc(eight->type, offs, eight);

    expr2tc shl = shl2tc(offs->type, effs, offs);
    expr2tc noteffs = bitnot2tc(effs->type, shl);
    source = bitand2tc(source->type, source, noteffs);

    expr2tc shl2 = shl2tc(offs->type, update, offs);
    expr2tc e = bitor2tc(offs->type, shl2, source);

    if (org_type)
      e = bitcast2tc(org_type, e);

    return convert_ast(e);
  }

  // We are merging two values: an 8 bit update value, and a larger source
  // value that we will have to merge it into. Start off by collecting
  // information about the source values and their widths.
  assert(
    is_number_type(data.source_value->type) &&
    "Byte update of unsupported data type");

  smt_astt value = convert_ast(data.update_value);
  smt_astt src_value = convert_ast(data.source_value);

  unsigned int width_op0 = data.source_value->type->get_width();
  unsigned int src_offset =
    to_constant_int2t(data.source_offset).value.to_uint64();

  // Flip location if we're in big-endian mode
  if (data.big_endian)
  {
    unsigned int data_size =
      type_byte_size(data.source_value->type).to_uint64() - 1;
    src_offset = data_size - src_offset;
  }

// Assertion some of our assumptions, which broadly mean that we'll only work
// on bytes that are going into non-byte words
#ifndef NDEBUG
  unsigned int width_op2 = data.update_value->type->get_width();
  assert(width_op2 == 8 && "Can't byte update non-byte operations");
  assert(width_op2 != width_op0 && "Can't byte update bytes, sorry");
#endif

  // Bail if this is an invalid update. This might be legitimate, in that one
  // can update a padding byte in a struct, leading to a crazy out of bounds
  // update. Either way, leave it to the dereference layer to decide on
  // invalidity.
  if (src_offset >= (width_op0 / 8))
    return convert_ast(data.source_value);

  // Build in three parts: the most significant bits, any in the middle, and
  // the bottom, of the reconstructed / merged output. There might not be a
  // middle if the update byte is at the top or the bottom.
  unsigned int top_of_update = (8 * src_offset) + 8;
  unsigned int bottom_of_update = (8 * src_offset);

  smt_astt top;
  if (top_of_update == width_op0)
  {
    top = value;
  }
  else
    top = mk_extract(src_value, width_op0 - 1, top_of_update);

  smt_astt middle;
  if (top == value)
  {
    middle = nullptr;
  }
  else
  {
    middle = value;
  }

  smt_astt bottom;
  if (src_offset == 0)
  {
    middle = nullptr;
    bottom = value;
  }
  else
    bottom = mk_extract(src_value, bottom_of_update - 1, 0);

  // Concatenate the top and bottom, and possible middle, together.
  smt_astt concat;

  if (middle != nullptr)
    concat = mk_concat(top, middle);
  else
    concat = top;

  return mk_concat(concat, bottom);
}
