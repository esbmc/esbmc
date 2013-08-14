#include "smt_conv.h"

const smt_ast *
smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);

  if (!is_constant_int2t(data.source_offset)) {
    std::cerr << "byte_extract expects constant 2nd arg";
    abort();
  }

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  unsigned width;
  width = data.source_value->type->get_width();
  // XXXjmorse - looks like this only ever reads a single byte, not the desired
  // number of bytes to fill the type.

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
    }

    unsigned int sort_sz = data.source_value->type->get_width();
    if (sort_sz < upper) {
      // Extends past the end of this data item. Should be fixed in some other
      // dedicated feature branch, in the meantime stop Z3 from crashing
      const smt_sort *s = mk_sort(SMT_SORT_BV, 8, false);
      return mk_smt_symbol("out_of_bounds_byte_extract", s);
    } else {
      return mk_extract(source, upper, lower, convert_sort(expr->type));
    }
  }

  std::cerr << "Unsupported byte extract operand" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_byte_update(const expr2tc &expr)
{
  const byte_update2t &data = to_byte_update2t(expr);

  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  if (!is_constant_int2t(data.source_offset)) {
    std::cerr << "byte_extract expects constant 2nd arg";
    abort();
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
  }

  std::cerr << "unsupported irep for convert_byte_update" << std::endl;;
  abort();
}

