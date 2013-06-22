#include <sstream>

#include <base_type.h>

#include "smt_conv.h"

const smt_ast *
smt_convt::convert_typecast_bool(const typecast2t &cast)
{

  if (is_bv_type(cast.from)) {
    notequal2tc neq(cast.from, zero_uint);
    return convert_ast(neq);
  } else if (is_pointer_type(cast.from)) {
    // Convert to two casts.
    typecast2tc to_int(machine_ptr, cast.from);
    constant_int2tc zero(machine_ptr, BigInt(0));
    equality2tc as_bool(zero, to_int);
    return convert_ast(as_bool);
  } else {
    std::cerr << "Unimplemented bool typecast" << std::endl;
    abort();
  }
}

const smt_ast *
smt_convt::convert_typecast_fixedbv_nonint(const expr2tc &expr)
{
  const smt_ast *args[4];
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;

  if (is_pointer_type(cast.from)) {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  const smt_ast *a = convert_ast(cast.from);
  const smt_sort *s = convert_sort(cast.type);

  if (is_bv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_integer_bits) {
      // Just concat fraction ozeros at the bottom
      args[0] = a;
    } else if (from_width > to_integer_bits) {
      const smt_sort *tmp = mk_sort(SMT_SORT_BV, from_width - to_integer_bits,
                                    false);
      args[0] = mk_extract(a, to_integer_bits-1, 0, tmp);
    } else {
      assert(from_width < to_integer_bits);
      const smt_sort *tmp = mk_sort(SMT_SORT_BV, to_integer_bits, false);
      args[0] = convert_sign_ext(a, tmp, from_width,
                                 to_integer_bits - from_width);
    }

    // Make all zeros fraction bits
    args[1] = mk_smt_bvint(BigInt(0), false, to_fraction_bits);
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  } else if (is_bool_type(cast.from)) {
    const smt_ast *args[3];
    const smt_sort *intsort;
    args[0] = a;
    args[1] = mk_smt_bvint(BigInt(0), false, to_integer_bits);
    args[2] = mk_smt_bvint(BigInt(1), false, to_integer_bits);
    intsort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
    args[0] = mk_func_app(intsort, SMT_FUNC_ITE, args, 3);
    args[1] = mk_smt_bvint(BigInt(0), false, to_integer_bits);
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  } else if (is_fixedbv_type(cast.from)) {
    // FIXME: conversion here for to_int_bits > from_int_bits is factually
    // broken, run 01_cbmc_Fixedbv8 with --no-simplify
    const smt_ast *magnitude, *fraction;

    const fixedbv_type2t &from_fbvt = to_fixedbv_type(cast.from->type);

    unsigned from_fraction_bits = from_fbvt.width - from_fbvt.integer_bits;
    unsigned from_integer_bits = from_fbvt.integer_bits;
    unsigned from_width = from_fbvt.width;

    if (to_integer_bits <= from_integer_bits) {
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
      magnitude = mk_extract(a, (from_fraction_bits + to_integer_bits - 1),
                             from_fraction_bits, tmp_sort);
    } else   {
      assert(to_integer_bits > from_integer_bits);
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV,
                                        from_width - from_fraction_bits, false);
      const smt_ast *ext = mk_extract(a, from_width - 1, from_fraction_bits,
                                      tmp_sort);

      tmp_sort = mk_sort(SMT_SORT_BV, (from_width - from_fraction_bits)
                                      + (to_integer_bits - from_integer_bits),
                                      false);
      magnitude = convert_sign_ext(ext, tmp_sort,
                                   from_width - from_fraction_bits,
                                   to_integer_bits - from_integer_bits);
    }

    if (to_fraction_bits <= from_fraction_bits) {
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
      fraction = mk_extract(a, from_fraction_bits - 1,
                            from_fraction_bits - to_fraction_bits, tmp_sort);
    } else {
      const smt_ast *args[2];
      assert(to_fraction_bits > from_fraction_bits);
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, from_fraction_bits,
                                         false);
      args[0] = mk_extract(a, from_fraction_bits -1, 0, tmp_sort);
      args[1] = mk_smt_bvint(BigInt(0), false,
                             to_fraction_bits - from_fraction_bits);

      tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
      fraction = mk_func_app(tmp_sort, SMT_FUNC_CONCAT, args, 2);
    }

    const smt_ast *args[2];
    args[0] = magnitude;
    args[1] = fraction;
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  }

  std::cerr << "unexpected typecast to fixedbv" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_typecast_to_ints(const typecast2t &cast)
{
  unsigned to_width = cast.type->get_width();
  const smt_sort *s = convert_sort(cast.type);
  const smt_ast *a = convert_ast(cast.from);

  if (is_signedbv_type(cast.from) || is_fixedbv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      if (int_encoding && is_signedbv_type(cast.from) &&
               is_fixedbv_type(cast.type)) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding && is_fixedbv_type(cast.from) &&
               is_signedbv_type(cast.type)) {
        return round_real_to_int(a);
      } else if (int_encoding && is_unsignedbv_type(cast.from) &&
                 is_signedbv_type(cast.type)) {
        // Unsigned -> Signed. Seeing how integer mode is an approximation,
        // just return the original value, and if it would have wrapped around,
        // too bad.
        return convert_ast(cast.from);
      } else if (int_encoding && is_signedbv_type(cast.from) &&
                 is_unsignedbv_type(cast.type)) {
        // XXX XXX XXX seriously rethink what this code attempts to do,
        // implementing something that tries to look like twos compliment.
        constant_int2tc maxint(cast.type, BigInt(0xFFFFFFFF));
        add2tc add(cast.type, maxint, cast.from);

        constant_int2tc zero(cast.from->type, BigInt(0));
        lessthan2tc lt(cast.from, zero);
        if2tc ite(cast.type, lt, add, cast.from);
        return convert_ast(ite);
      } else if (!int_encoding && is_fixedbv_type(cast.from) &&
                 is_bv_type(cast.type)) {
        return round_fixedbv_to_int(a, from_width, to_width);
      } else if ((is_signedbv_type(cast.type) && is_unsignedbv_type(cast.from))
            || (is_unsignedbv_type(cast.type) && is_signedbv_type(cast.from))) {
        // Operands have differing signs (and same width). Just return.
        return convert_ast(cast.from);
      } else {
        std::cerr << "Unrecognized equal-width int typecast format" <<std::endl;
        abort();
      }
    } else if (from_width < to_width) {
      if (int_encoding &&
          ((is_fixedbv_type(cast.type) && is_signedbv_type(cast.from)))) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding) {
	return a; // output = output
      } else {
        return convert_sign_ext(a, s, from_width, (to_width - from_width));
      }
    } else if (from_width > to_width) {
      if (int_encoding &&
          ((is_signedbv_type(cast.from) && is_fixedbv_type(cast.type)))) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding &&
                (is_fixedbv_type(cast.from) && is_signedbv_type(cast.type))) {
        return round_real_to_int(a);
      } else if (int_encoding) {
        return a; // output = output
      } else {
	if (!to_width)
          to_width = config.ansi_c.int_width;

        return mk_extract(a, to_width-1, 0, s);
      }
    }
  } else if (is_unsignedbv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      return a; // output = output
    } else if (from_width < to_width) {
      if (int_encoding) {
	return a; // output = output
      } else {
        return convert_zero_ext(a, s, (to_width - from_width));
      }
    } else if (from_width > to_width) {
      if (int_encoding) {
	return a; // output = output
      } else {
        return mk_extract(a, to_width - 1, 0, s);
      }
    }
  } else if (is_bool_type(cast.from)) {
    const smt_ast *zero, *one;
    unsigned width = cast.type->get_width();

    if (is_bv_type(cast.type)) {
      if (int_encoding) {
        zero = mk_smt_int(BigInt(0), false);
        one = mk_smt_int(BigInt(1), false);
      } else {
        zero = mk_smt_bvint(BigInt(0), false, width);
        one = mk_smt_bvint(BigInt(1), false, width);
      }
    } else if (is_fixedbv_type(cast.type)) {
      zero = mk_smt_real("0");
      one = mk_smt_real("1");
    } else {
      std::cerr << "Unexpected type in typecast of bool" << std::endl;
      abort();
    }

    const smt_ast *args[3];
    args[0] = a;
    args[1] = one;
    args[2] = zero;
    return mk_func_app(s, SMT_FUNC_ITE, args, 3);
  }

  std::cerr << "Unexpected type in int/ptr typecast" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_typecast_to_ptr(const typecast2t &cast)
{

  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if (is_pointer_type(cast.from)) {
    return convert_ast(cast.from);
  }

  // Unpleasentness; we don't know what pointer this integer is going to
  // correspond to, and there's no way of telling statically, so we have
  // to enumerate all pointers it could point at. IE, all of them. Which
  // is expensive, but here we are.

  // First cast it to an unsignedbv
  type2tc int_type = machine_uint;
  typecast2tc cast_to_unsigned(int_type, cast.from);
  expr2tc target = cast_to_unsigned;

  // Construct array for all possible object outcomes
  expr2tc is_in_range[addr_space_data.back().size()];
  expr2tc obj_ids[addr_space_data.back().size()];
  expr2tc obj_starts[addr_space_data.back().size()];

  std::map<unsigned,unsigned>::const_iterator it;
  unsigned int i;
  for (it = addr_space_data.back().begin(), i = 0;
       it != addr_space_data.back().end(); it++, i++)
  {
    unsigned id = it->first;
    obj_ids[i] = constant_int2tc(int_type, BigInt(id));

    std::stringstream ss1, ss2;
    ss1 << "__ESBMC_ptr_obj_start_" << id;
    symbol2tc ptr_start(int_type, ss1.str());
    ss2 << "__ESBMC_ptr_obj_end_" << id;
    symbol2tc ptr_end(int_type, ss2.str());

    obj_starts[i] = ptr_start;

    greaterthanequal2tc ge(target, ptr_start);
    lessthanequal2tc le(target, ptr_end);
    and2tc theand(ge, le);
    is_in_range[i] = theand;
  }

  // Generate a big ITE chain, selecing a particular pointer offset. A
  // significant question is what happens when it's neither; in which case I
  // suggest the ptr becomes invalid_object. However, this needs frontend
  // support to check for invalid_object after all dereferences XXXjmorse.

  // So, what's the default value going to be if it doesn't match any existing
  // pointers? Answer, it's going to be the invalid object identifier, but with
  // an offset that calculates to the integer address of this object.
  // That's so that we can store an invalid pointer in a pointer type, that
  // eventually can be converted back via some mechanism to a valid pointer.
  expr2tc id, offs;
  id = constant_int2tc(int_type, pointer_logic.back().get_invalid_object());

  // Calculate ptr offset - target minus start of invalid range, ie 1
  offs = sub2tc(int_type, target, one_uint);

  std::vector<expr2tc> membs;
  membs.push_back(id);
  membs.push_back(offs);
  expr2tc prev_in_chain = constant_struct2tc(pointer_struct, membs);

  // Now that big ite chain,
  for (i = 0; i < addr_space_data.back().size(); i++) {
    membs.clear();

    // Calculate ptr offset were it this
    offs = sub2tc(int_type, target, obj_starts[i]);

    membs.push_back(obj_ids[i]);
    membs.push_back(offs);
    constant_struct2tc selected_tuple(pointer_struct, membs);

    prev_in_chain = if2tc(pointer_struct, is_in_range[i],
                          selected_tuple, prev_in_chain);
  }

  // Finally, we're now at the point where prev_in_chain represents a pointer
  // object. Hurrah.
  return convert_ast(prev_in_chain);
}

const smt_ast *
smt_convt::convert_typecast_from_ptr(const typecast2t &cast)
{

  type2tc int_type = machine_uint;

  // The plan: index the object id -> address-space array and pick out the
  // start address, then add it to any additional pointer offset.

  pointer_object2tc obj_num(int_type, cast.from);

  symbol2tc addrspacesym(addr_space_arr_type, get_cur_addrspace_ident());
  index2tc idx(addr_space_type, addrspacesym, obj_num);

  // We've now grabbed the pointer struct, now get first element. Represent
  // as fetching the first element of the struct representation.
  member2tc memb(int_type, idx, addr_space_type_data->member_names[0]);

  pointer_offset2tc ptr_offs(int_type, cast.from);
  add2tc add(int_type, memb, ptr_offs);

  // Finally, replace typecast
  typecast2tc new_cast(cast.type, add);
  return convert_ast(new_cast);
}

const smt_ast *
smt_convt::convert_typecast_struct(const typecast2t &cast)
{

  const struct_type2t &struct_type_from = to_struct_type(cast.from->type);
  const struct_type2t &struct_type_to = to_struct_type(cast.type);

  u_int i = 0, i2 = 0;

  std::vector<type2tc> new_members;
  std::vector<irep_idt> new_names;
  new_members.reserve(struct_type_to.members.size());
  new_names.reserve(struct_type_to.members.size());

  i = 0;
  // This all goes to pot when we consider polymorphism, and in particular,
  // multiple inheritance. So, for normal structs, as usual check that each
  // field has a compatible type. But for classes, check that either they're
  // the same class, or the source is a subclass of the target type. If so,
  // we just select out the common fields, which drops any additional data in
  // the subclass.

  bool same_format = true;
  if (is_subclass_of(cast.from->type, cast.type, ns)) {
    same_format = false; // then we're fine
  } else if (struct_type_from.name == struct_type_to.name) {
    ; // Also fine
  } else {
    // Check that these two different structs have the same format.
    forall_types(it, struct_type_to.members) {
      if (!base_type_eq(struct_type_from.members[i], *it, ns)) {
        std::cerr << "Incompatible struct in cast-to-struct" << std::endl;
        abort();
      }

      i++;
    }
  }

  smt_sort *fresh_sort = convert_sort(cast.type);
  smt_ast *fresh = tuple_fresh(fresh_sort);
  const smt_ast *src_ast = convert_ast(cast.from);
  smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  if (same_format) {
    // Alas, Z3 considers field names as being part of the type, so we can't
    // just consider the source expression to be the casted expression.
    i2 = 0;
    forall_types(it, struct_type_to.members) {
      const smt_ast *args[2];
      smt_sort *this_sort = convert_sort(*it);
      args[0] = tuple_project(src_ast, this_sort, i2);
      args[1] = tuple_project(fresh, this_sort, i2);
      const smt_ast *eq;
      if (is_struct_type(*it) || is_union_type(*it) || is_pointer_type(*it))
        eq = tuple_equality(args[0], args[1]);
      else
        eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      assert_lit(mk_lit(eq));
      i2++;
    }
  } else {
    // Due to inheritance, these structs don't have the same format. Therefore
    // we have to look up source fields by matching the field names between
    // structs, then using their index numbers construct equalities between
    // fields in the source value and a fresh value.
    i2 = 0;
    forall_names(it, struct_type_to.member_names) {
      // Linear search, yay :(
      unsigned int i3 = 0;
      forall_names(it2, struct_type_from.member_names) {
        if (*it == *it2)
          break;
        i3++;
      }

      assert(i3 != struct_type_from.member_names.size() &&
             "Superclass field doesn't exist in subclass during conversion "
             "cast");
      // Could assert that the types are the same, however Z3 is going to
      // complain mightily if we get it wrong.

      const smt_ast *args[2];
      const type2tc &thetype = struct_type_from.members[i3];
      smt_sort *this_sort = convert_sort(thetype);
      args[0] = tuple_project(src_ast, this_sort, i3);
      args[1] = tuple_project(fresh, this_sort, i2);

      const smt_ast *eq;
      if (is_struct_type(thetype) || is_union_type(thetype) ||
          is_pointer_type(thetype))
        eq = tuple_equality(args[0], args[1]);
      else
        eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      assert_lit(mk_lit(eq));
      i2++;
    }
   }

  return fresh;
}

const smt_ast *
smt_convt::convert_typecast(const expr2tc &expr)
{

  const typecast2t &cast = to_typecast2t(expr);
  if (cast.type == cast.from->type)
    return convert_ast(cast.from);

  if (is_pointer_type(cast.type)) {
    return convert_typecast_to_ptr(cast);
  } else if (is_pointer_type(cast.from)) {
    return convert_typecast_from_ptr(cast);
  } else if (is_bool_type(cast.type)) {
    return convert_typecast_bool(cast);
  } else if (is_fixedbv_type(cast.type) && !int_encoding)      {
    return convert_typecast_fixedbv_nonint(expr);
  } else if (is_bv_type(cast.type) ||
             is_fixedbv_type(cast.type) ||
             is_pointer_type(cast.type)) {
    return convert_typecast_to_ints(cast);
  } else if (is_struct_type(cast.type))     {
    return convert_typecast_struct(cast);
  } else if (is_union_type(cast.type)) {
    if (base_type_eq(cast.type, cast.from->type, namespacet(contextt()))) {
      return convert_ast(cast.from); // No additional conversion required
    } else {
      std::cerr << "Can't typecast between unions" << std::endl;
      abort();
    }
  }

  // XXXjmorse -- what about all other types, eh?
  std::cerr << "Typecast for unexpected type" << std::endl;
  abort();
}

