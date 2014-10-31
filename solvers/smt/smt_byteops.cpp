#include "smt_conv.h"

smt_astt 
smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);

  assert(is_scalar_type(data.source_value) && "Byte extract now only works on "
         "scalar variables");
  if (!is_constant_int2t(data.source_offset)) {
    expr2tc source = data.source_value;
    unsigned int src_width = source->type->get_width();
    if (!is_bv_type(source)) {
      source = typecast2tc(get_uint_type(src_width), source);
    }

    // The approach: the argument is now a bitvector. Just shift it the
    // appropriate amount, according to the source offset, and select out the
    // bottom byte.
    expr2tc offs = data.source_offset;
    if (offs->type->get_width() != src_width)
      // Z3 requires these two arguments to be the same width
      offs = typecast2tc(source->type, data.source_offset);

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
    upper = ((intref.constant_value.to_long() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.constant_value.to_long() * 8; //i*w;
  } else {
    uint64_t max = width - 1;
    upper = max - (intref.constant_value.to_long() * 8); //max-(i*w);
    lower = max - ((intref.constant_value.to_long() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  smt_astt source = convert_ast(data.source_value);;

  if (int_encoding) {
    std::cerr << "Refusing to byte extract in integer mode; re-run in "
                 "bitvector mode" << std::endl;
    abort();
  } else {
    if (is_bv_type(data.source_value)) {
      ;
    } else if (is_fixedbv_type(data.source_value)) {
      ;
    } else {
      std::cerr << "Unrecognized type in operand to byte extract." << std::endl;
      data.dump();
      abort();
    }

    unsigned int sort_sz = data.source_value->type->get_width();
    if (sort_sz <= upper) {
      smt_sortt s = mk_sort(SMT_SORT_BV, 8, false);
      return mk_smt_symbol("out_of_bounds_byte_extract", s);
    } else {
      return mk_extract(source, upper, lower, convert_sort(expr->type));
    }
  }
}

smt_astt 
smt_convt::convert_byte_update(const expr2tc &expr)
{
  const byte_update2t &data = to_byte_update2t(expr);

  assert(is_scalar_type(data.source_value) && "Byte update only works on "
         "scalar variables now");

  if (!is_constant_int2t(data.source_offset)) {
    if (is_pointer_type(data.type)) {
      // Just return a free pointer. Seriously, this is going to be faster,
      // easier, and probably accurate than anything else.
      smt_sortt s = convert_sort(data.type);
      return mk_fresh(s, "updated_ptr");
    }

    expr2tc source = data.source_value;
    unsigned int src_width = source->type->get_width();
    if (!is_bv_type(source))
      source = typecast2tc(get_uint_type(src_width), source);

    expr2tc offs = data.source_offset;
    if (offs->type->get_width() != src_width)
      offs = typecast2tc(get_uint_type(src_width), offs);

    expr2tc update = data.update_value;
    if (update->type->get_width() != src_width)
      update = typecast2tc(get_uint_type(src_width), update);

    // The approach: mask, shift and or. XXX, byte order?
    // Massively inefficient.

    expr2tc eight = constant_int2tc(get_uint_type(src_width), BigInt(8));
    expr2tc effs = constant_int2tc(eight->type, BigInt(255));
    offs = mul2tc(eight->type, offs, eight);

    expr2tc shl = shl2tc(offs->type, effs, offs);
    expr2tc noteffs = bitnot2tc(effs->type, shl);
    source = bitand2tc(source->type, source, noteffs);

    expr2tc shl2 = shl2tc(offs->type, update, offs);
    return convert_ast(bitor2tc(offs->type, shl2, source));
  }

  smt_astt value;
  unsigned int width_op0, width_op2;

  value = convert_ast(data.update_value);

  width_op2 = data.update_value->type->get_width();

  if (int_encoding) {
    std::cerr << "Can't byte update in integer mode; rerun in bitvector mode"
              << std::endl;
    abort();
  }

  if (is_signedbv_type(data.source_value->type)) {
    width_op0 = data.source_value->type->get_width();

    if (width_op0 == 0) {
      // XXXjmorse - can this ever happen now?
      std::cerr << "failed to get width of byte_update operand";
      abort();
    }

    if (width_op0 > width_op2) {
      return convert_sign_ext(value, convert_sort(expr->type), width_op2,
                              width_op0 - width_op2);
    } else if (width_op0 == width_op2 &&
        to_constant_int2t(data.source_offset).constant_value.to_ulong() == 0) {
      // Byte update at offset zero with value of same size. Just return update
      // value. Ideally this shouldn't ever be encoded, but it needn't be fatal
      // if it is, just unperformant.
      return convert_ast(data.update_value);
    } else {
      std::cerr << "unsupported irep for conver_byte_update" << std::endl;
      abort();
    }
  } else if (is_unsignedbv_type(data.source_value->type)) {
    width_op0 = data.source_value->type->get_width();
    assert(width_op0 != 0);

    if (width_op0 > width_op2) {
      return convert_zero_ext(value, convert_sort(expr->type),
                              width_op0 - width_op2);
    } else {
      std::cerr << "unsupported irep for conver_byte_update" << std::endl;
      abort();
    }
  }

  std::cerr << "unsupported irep for convert_byte_update" << std::endl;;
  abort();
}
