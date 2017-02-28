#include "smt_conv.h"

smt_astt
smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);
  expr2tc source = data.source_value;
  unsigned int src_width = source->type->get_width();

  if (!is_bv_type(source->type) && !is_fixedbv_type(source->type))
    source = bitcast2tc(get_uint_type(src_width), source);

  assert(is_scalar_type(data.source_value) && "Byte extract now only works on "
         "scalar variables");
  if (!is_constant_int2t(data.source_offset)) {
    // The approach: the argument is now a bitvector. Just shift it the
    // appropriate amount, according to the source offset, and select out the
    // bottom byte.
    expr2tc offs = data.source_offset;

    // Endian-ness: if we're in non-"native" endian-ness mode, then flip the
    // offset distance. The rest of these calculations will still apply.
    if (data.big_endian) {
      auto data_size = type_byte_size(source->type);
      constant_int2tc data_size_expr(source->type, data_size - 1);
      sub2tc sub(source->type, data_size_expr, offs);
      offs = sub;
    }

    if (offs->type->get_width() != src_width)
      // Z3 requires these two arguments to be the same width
      offs = typecast2tc(source->type, data.source_offset);

    offs = mul2tc(offs->type, offs, gen_ulong(8));

    lshr2tc shr(source->type, source, offs);
    smt_astt ext = convert_ast(shr);
    smt_astt res = mk_extract(ext, 7, 0, convert_sort(get_uint8_type()));
    return res;
  }

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  unsigned width;
  width = data.source_value->type->get_width();

  uint64_t upper, lower;
  if (!data.big_endian) {
    upper = ((intref.value.to_long() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.value.to_long() * 8; //i*w;
  } else {
    uint64_t max = width - 1;
    upper = max - (intref.value.to_long() * 8); //max-(i*w);
    lower = max - ((intref.value.to_long() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  smt_astt source_ast = convert_ast(source);

  if (int_encoding) {
    std::cerr << "Refusing to byte extract in integer mode; re-run in "
                 "bitvector mode" << std::endl;
    abort();
  } else {
    unsigned int sort_sz = data.source_value->type->get_width();
    if (sort_sz <= upper) {
      smt_sortt s = mk_sort(SMT_SORT_BV, 8, false);
      return mk_smt_symbol("out_of_bounds_byte_extract", s);
    } else {
      return mk_extract(source_ast, upper, lower, convert_sort(expr->type));
    }
  }
}

smt_astt
smt_convt::convert_byte_update(const expr2tc &expr)
{
  const byte_update2t &data = to_byte_update2t(expr);

  assert(is_scalar_type(data.source_value) && "Byte update only works on "
         "scalar variables now");
  assert(data.type == data.source_value->type);

  if (!is_bv_type(data.type) && !is_fixedbv_type(data.type)) {
    // This is a pointer or a bool, or something. We don't want to handle
    // casting of it in the body of this function, so wrap it up as a bitvector
    // and re-apply.
    type2tc bit_type = get_uint_type(data.type->get_width());
    bitcast2tc src_obj(bit_type, data.source_value);
    byte_update2tc new_update(bit_type, src_obj, data.source_offset,
        data.update_value, data.big_endian);
    bitcast2tc cast_back(data.type, new_update);
    return convert_ast(cast_back);
  }

  if (!is_constant_int2t(data.source_offset)) {
    expr2tc source = data.source_value;
    unsigned int src_width = source->type->get_width();
    if (!is_bv_type(source))
      source = typecast2tc(get_uint_type(src_width), source);

    expr2tc offs = data.source_offset;

    // Endian-ness: if we're in non-"native" endian-ness mode, then flip the
    // offset distance. The rest of these calculations will still apply.
    if (data.big_endian) {
      auto data_size = type_byte_size(source->type);
      constant_int2tc data_size_expr(source->type, data_size - 1);
      sub2tc sub(source->type, data_size_expr, offs);
      offs = sub;
    }

    if (offs->type->get_width() != src_width)
      offs = typecast2tc(get_uint_type(src_width), offs);

    expr2tc update = data.update_value;
    if (update->type->get_width() != src_width)
      update = typecast2tc(get_uint_type(src_width), update);

    // The approach: mask, shift and or. Quite inefficient.

    expr2tc eight = constant_int2tc(get_uint_type(src_width), BigInt(8));
    expr2tc effs = constant_int2tc(eight->type, BigInt(255));
    offs = mul2tc(eight->type, offs, eight);

    expr2tc shl = shl2tc(offs->type, effs, offs);
    expr2tc noteffs = bitnot2tc(effs->type, shl);
    source = bitand2tc(source->type, source, noteffs);

    expr2tc shl2 = shl2tc(offs->type, update, offs);
    return convert_ast(bitor2tc(offs->type, shl2, source));
  }

  // We are merging two values: an 8 bit update value, and a larger source
  // value that we will have to merge it into. Start off by collecting
  // information about the source values and their widths.
  assert(is_number_type(data.source_value->type) && "Byte update of unsupported data type");

  smt_astt value, src_value;
  unsigned int width_op0, src_offset;

  value = convert_ast(data.update_value);
  src_value = convert_ast(data.source_value);

  width_op0 = data.source_value->type->get_width();
  src_offset = to_constant_int2t(data.source_offset).value.to_ulong();

  // Flip location if we're in big-endian mode
  if (data.big_endian) {
    unsigned int data_size =
      type_byte_size(data.source_value->type).to_ulong() - 1;
    src_offset = data_size - src_offset;
  }

  if (int_encoding) {
    std::cerr << "Can't byte update in integer mode; rerun in bitvector mode"
              << std::endl;
    abort();
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

  smt_astt top, middle, bottom;

  // Build in three parts: the most significant bits, any in the middle, and
  // the bottom, of the reconstructed / merged output. There might not be a
  // middle if the update byte is at the top or the bottom.
  unsigned int top_of_update = (8 * src_offset) + 8;
  unsigned int bottom_of_update = (8 * src_offset);

  if (top_of_update == width_op0) {
    top = value;
  } else {
    smt_sortt s = mk_sort(SMT_SORT_BV, width_op0 - top_of_update, false);
    top = mk_extract(src_value, width_op0 - 1, top_of_update, s);
  }

  if (top == value) {
    middle = NULL;
  } else {
    middle = value;
  }

  if (src_offset == 0) {
    middle = NULL;
    bottom = value;
  } else {
    smt_sortt s = mk_sort(SMT_SORT_BV, bottom_of_update, false);
    bottom = mk_extract(src_value, bottom_of_update - 1, 0, s);
  }

  // Concatenate the top and bottom, and possible middle, together.
  smt_astt concat;

  if (middle != NULL) {
    smt_sortt s = mk_sort(SMT_SORT_BV, width_op0 - bottom_of_update, false);
    concat = mk_func_app(s, SMT_FUNC_CONCAT, top, middle);
  } else {
    concat = top;
  }

  return mk_func_app(src_value->sort, SMT_FUNC_CONCAT, concat, bottom);
}
