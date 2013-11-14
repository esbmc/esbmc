#include "smt_conv.h"

const smt_ast *
smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);

  if (!is_constant_int2t(data.source_offset)) {
    assert(!is_structure_type(data.source_value) &&
           !is_array_type(data.source_value) && "Composite typed argument to "
           "byte extract");

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
    const smt_ast *ext = convert_ast(shr);
    const smt_ast *res = mk_extract(ext, 7, 0, convert_sort(get_uint8_type()));
    return res;
  }

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  unsigned width;
  try {
    width = data.source_value->type->get_width();
  } catch (array_type2t::dyn_sized_array_excp *p) {
    // Dynamically sized array. How to handle -- for now, assume that it's a
    // byte array, and select the relevant portions out.
    const array_type2t &arr_type = to_array_type(data.source_value->type);
    assert(is_scalar_type(arr_type.subtype) && "Can't cope with dynamic "
           "nonscalar arrays right now, sorry");

    expr2tc src_offs = data.source_offset;
    expr2tc expr = index2tc(arr_type.subtype, data.source_value, src_offs);

    if (!is_number_type(arr_type.subtype))
      expr = typecast2tc(get_uint8_type(), expr);

    return convert_ast(expr);
  }

  uint64_t upper, lower;
  if (!data.big_endian) {
    upper = ((intref.constant_value.to_long() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.constant_value.to_long() * 8; //i*w;
  } else {
    uint64_t max = width - 1;
    upper = max - (intref.constant_value.to_long() * 8); //max-(i*w);
    lower = max - ((intref.constant_value.to_long() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  const smt_ast *source = convert_ast(data.source_value);;

  if (int_encoding) {
    std::cerr << "Refusing to byte extract in integer mode; re-run in "
                 "bitvector mode" << std::endl;
    abort();
  } else {
    if (is_struct_type(data.source_value)) {
      const struct_type2t &struct_type =to_struct_type(data.source_value->type);
      unsigned i = 0, num_elems = struct_type.members.size();
      const smt_ast *struct_elem[num_elems + 1], *struct_elem_inv[num_elems +1];

      forall_types(it, struct_type.members) {
        struct_elem[i] = tuple_project(source, convert_sort(*it), i);
        i++;
      }

      for (unsigned k = 0; k < num_elems; k++)
        struct_elem_inv[(num_elems - 1) - k] = struct_elem[k];

      // Concat into one massive vector.
      const smt_ast *args[2];
      for (unsigned k = 0; k < num_elems; k++)
      {
        if (k == 1) {
          args[0] = struct_elem_inv[k - 1];
          args[1] = struct_elem_inv[k];
          // FIXME: sorts
          struct_elem_inv[num_elems] = mk_func_app(NULL, SMT_FUNC_CONCAT, args,
                                                   2);
        } else if (k > 1) {
          args[0] = struct_elem_inv[num_elems];
          args[1] = struct_elem_inv[k];
          // FIXME: sorts
          struct_elem_inv[num_elems] = mk_func_app(NULL, SMT_FUNC_CONCAT, args,
                                                   2);
        }
      }

      source = struct_elem_inv[num_elems];
    } else if (is_bv_type(data.source_value)) {
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
      const smt_sort *s = mk_sort(SMT_SORT_BV, 8, false);
      return mk_smt_symbol("out_of_bounds_byte_extract", s);
    } else {
      return mk_extract(source, upper, lower, convert_sort(expr->type));
    }
  }
}

const smt_ast *
smt_convt::convert_byte_update(const expr2tc &expr)
{
  const byte_update2t &data = to_byte_update2t(expr);

  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  if (!is_constant_int2t(data.source_offset)) {
    if (is_pointer_type(data.type)) {
      // Just return a free pointer. Seriously, this is going to be faster,
      // easier, and probably accurate than anything else.
      const smt_sort *s = convert_sort(data.type);
      return mk_fresh(s, "updated_ptr");
    }

    assert(!is_structure_type(data.source_value) &&
           !is_array_type(data.source_value) && "Composite typed argument to "
           "byte update");

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

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  const smt_ast *tuple, *value;
  unsigned int width_op0, width_op2;

  tuple = convert_ast(data.source_value);
  value = convert_ast(data.update_value);

  width_op2 = data.update_value->type->get_width();

  if (int_encoding) {
    std::cerr << "Can't byte update in integer mode; rerun in bitvector mode"
              << std::endl;
    abort();
  }

  if (is_struct_type(data.source_value)) {
    const struct_type2t &struct_type = to_struct_type(data.source_value->type);
    bool has_field = false;

    // XXXjmorse, this isn't going to be the case if it's a with.

    forall_types(it, struct_type.members) {
      width_op0 = (*it)->get_width();

      if (((*it)->type_id == data.update_value->type->type_id) &&
          (width_op0 == width_op2))
	has_field = true;
    }

    if (has_field)
      return tuple_update(tuple, intref.constant_value.to_long(),
                          data.update_value);
    else
      return tuple;
  } else if (is_signedbv_type(data.source_value->type)) {
    width_op0 = data.source_value->type->get_width();

    if (width_op0 == 0) {
      // XXXjmorse - can this ever happen now?
      std::cerr << "failed to get width of byte_update operand";
      abort();
    }

    if (width_op0 > width_op2) {
      return convert_sign_ext(value, convert_sort(expr->type), width_op2,
                              width_op0 - width_op2);
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
