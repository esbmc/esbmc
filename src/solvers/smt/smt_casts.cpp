#include <sstream>

#include <base_type.h>
#include <expr_util.h>

#include "smt_conv.h"

smt_astt
smt_convt::convert_typecast_to_bool(const typecast2t &cast)
{
  if (is_pointer_type(cast.from)) {
    // Convert to two casts.
    typecast2tc to_int(machine_ptr, cast.from);
    constant_int2tc zero(machine_ptr, BigInt(0));
    equality2tc as_bool(zero, to_int);
    return convert_ast(as_bool);
  }

  expr2tc zero_expr;
  migrate_expr(gen_zero(migrate_type_back(cast.from->type)), zero_expr);

  notequal2tc neq(cast.from, zero_expr);
  return convert_ast(neq);
}

smt_astt
smt_convt::convert_typecast_to_fixedbv_nonint(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);

  if (is_pointer_type(cast.from)) {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  if (is_bv_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_bv(expr);
  else if (is_bool_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_bool(expr);
  else if (is_fixedbv_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_fixedbv(expr);

  std::cerr << "unexpected typecast to fixedbv" << std::endl;
  abort();
}

smt_astt
smt_convt::convert_typecast_to_fixedbv_nonint_from_bv(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;
  assert(is_bv_type(cast.from));

  smt_astt a = convert_ast(cast.from);
  smt_sortt s = convert_sort(cast.type);

  unsigned from_width = cast.from->type->get_width();

  smt_astt frontpart;
  if (from_width == to_integer_bits) {
    // Just concat fraction ozeros at the bottom
    frontpart = a;
  } else if (from_width > to_integer_bits) {
    smt_sortt tmp = mk_sort(SMT_SORT_BV, from_width - to_integer_bits,
                            false);
    frontpart = mk_extract(a, to_integer_bits - 1, 0, tmp);
  } else {
    assert(from_width < to_integer_bits);
    smt_sortt tmp = mk_sort(SMT_SORT_BV, to_integer_bits, false);
    frontpart = convert_sign_ext(a, tmp, from_width,
                                 to_integer_bits - from_width);
  }

  // Make all zeros fraction bits
  smt_astt zero_fracbits = mk_smt_bvint(BigInt(0), false, to_fraction_bits);
  return mk_func_app(s, SMT_FUNC_CONCAT, frontpart, zero_fracbits);
}

smt_astt
smt_convt::convert_typecast_to_fixedbv_nonint_from_bool(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_integer_bits = fbvt.integer_bits;
  assert(is_bool_type(cast.from));

  smt_astt a = convert_ast(cast.from);
  smt_sortt s = convert_sort(cast.type);

  smt_sortt intsort;
  smt_astt zero = mk_smt_bvint(BigInt(0), false, to_integer_bits);
  smt_astt one = mk_smt_bvint(BigInt(1), false, to_integer_bits);
  intsort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
  smt_astt switched = mk_func_app(intsort, SMT_FUNC_ITE, a, zero, one);
  return mk_func_app(s, SMT_FUNC_CONCAT, switched, zero);
}

smt_astt
smt_convt::convert_typecast_to_fixedbv_nonint_from_fixedbv(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  assert(is_fixedbv_type(cast.from));
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  const fixedbv_type2t &from_fbvt = to_fixedbv_type(cast.from->type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;
  unsigned from_fraction_bits = from_fbvt.width - from_fbvt.integer_bits;
  unsigned from_integer_bits = from_fbvt.integer_bits;
  unsigned from_width = from_fbvt.width;
  smt_astt magnitude, fraction;
  smt_astt a = convert_ast(cast.from);
  smt_sortt s = convert_sort(cast.type);

  // FIXME: conversion here for to_int_bits > from_int_bits is factually
  // broken, run 01_cbmc_Fixedbv8 with --no-simplify

  // The plan here is to extract the magnitude and fraction from the source
  // fbv, extend or truncate them appropriately, then concatenate them.

  // Start with the magnitude
  if (to_integer_bits <= from_integer_bits) {
    smt_sortt tmp_sort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
    magnitude = mk_extract(a, (from_fraction_bits + to_integer_bits - 1),
                           from_fraction_bits, tmp_sort);
  } else {
    assert(to_integer_bits > from_integer_bits);
    smt_sortt tmp_sort = mk_sort(SMT_SORT_BV, from_integer_bits, false);
    smt_astt ext = mk_extract(a, from_width - 1, from_fraction_bits, tmp_sort);

    unsigned int additional_bits = to_integer_bits - from_integer_bits;
    tmp_sort = mk_sort(SMT_SORT_BV, from_integer_bits + additional_bits, false);
    magnitude = convert_sign_ext(ext, tmp_sort, from_integer_bits,
                                 additional_bits);
  }

  // Followed by the fraction part
  if (to_fraction_bits <= from_fraction_bits) {
    smt_sortt tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
    fraction = mk_extract(a, from_fraction_bits - 1,
                          from_fraction_bits - to_fraction_bits, tmp_sort);
  } else {
    assert(to_fraction_bits > from_fraction_bits);

    // Increase the size of the fraction by adding zeros on the end. This is
    // not a zero extension because they're at the end, not the start
    smt_sortt tmp_sort = mk_sort(SMT_SORT_BV, from_fraction_bits, false);
    smt_astt src_fraction = mk_extract(a, from_fraction_bits - 1, 0, tmp_sort);
    smt_astt zeros = mk_smt_bvint(BigInt(0), false,
                                  to_fraction_bits - from_fraction_bits);

    tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
    fraction = mk_func_app(tmp_sort, SMT_FUNC_CONCAT, src_fraction, zeros);
  }

  // Finally, concatenate the adjusted magnitude / fraction
  return mk_func_app(s, SMT_FUNC_CONCAT, magnitude, fraction);
}

smt_astt
smt_convt::convert_typecast_to_ints(const typecast2t &cast)
{

  if (int_encoding)
    return convert_typecast_to_ints_intmode(cast);

  if (is_signedbv_type(cast.from) || is_fixedbv_type(cast.from)) {
    return convert_typecast_to_ints_from_fbv_sint(cast);
  } else if (is_unsignedbv_type(cast.from)) {
    return convert_typecast_to_ints_from_unsigned(cast);
  } else if (is_floatbv_type(cast.from)) {
    return mk_smt_typecast_from_bvfloat(cast);
  } else if (is_bool_type(cast.from)) {
    return convert_typecast_to_ints_from_bool(cast);
  }

  std::cerr << "Unexpected type in int/ptr typecast" << std::endl;
  abort();
}

smt_astt
smt_convt::convert_typecast_to_ints_intmode(const typecast2t &cast)
{
  assert(int_encoding);
  // Integer-mode conversion of integers. Immediately, we don't care about
  // bit widths, to the extent that any fixedbv <=> fixedbv conversion can
  // remain a real, and any {un,}signedbv <=> {un,}signedbv conversion can
  // remain an int. The only thing we actually need to care about is the
  // conversion between ints and reals.

  if (is_fixedbv_type(cast.type) && is_fixedbv_type(cast.from))
    return convert_ast(cast.from);

  if (is_bv_type(cast.type) && is_bv_type(cast.from))
    // NB: this means negative numbers assigned to unsigned ints remain
    // negative. This IMO is one of the inaccuracies accepted by the use of
    // ir mode.
    return convert_ast(cast.from);

  smt_astt a = convert_ast(cast.from);

  // Handle conversions from booleans
  if (is_bool_type(cast.from)) {
    smt_astt zero, one;
    if (is_bv_type(cast.type)) {
      zero = mk_smt_int(BigInt(0), false);
      one = mk_smt_int(BigInt(1), false);
    } else {
      zero = mk_smt_real("0");
      one = mk_smt_real("1");
    }

    return mk_func_app(convert_sort(cast.type), SMT_FUNC_ITE, a, one, zero);
  }

  // Otherwise, we're looking at a cast between reals and int sorts.
  if (is_fixedbv_type(cast.type)) {
    assert(is_bv_type(cast.from));
    return mk_func_app(convert_sort(cast.type), SMT_FUNC_INT2REAL, &a, 1);
  } else {
    assert(is_bv_type(cast.type));
    assert(is_fixedbv_type(cast.from));
    return round_real_to_int(a);
  }
}

smt_astt
smt_convt::convert_typecast_to_ints_from_fbv_sint(const typecast2t &cast)
{
  assert(!int_encoding);
  unsigned to_width = cast.type->get_width();
  smt_sortt s = convert_sort(cast.type);
  smt_astt a = convert_ast(cast.from);

  unsigned from_width = cast.from->type->get_width();

  if (from_width == to_width) {
    if (is_fixedbv_type(cast.from) && is_bv_type(cast.type)) {
      return round_fixedbv_to_int(a, from_width, to_width);
    } else if ((is_signedbv_type(cast.type) && is_unsignedbv_type(cast.from))
               || (is_unsignedbv_type(cast.type) &&
                   is_signedbv_type(cast.from))) {
      // Operands have differing signs (and same width). Just return.
      return convert_ast(cast.from);
    } else {
      std::cerr << "Unrecognized equal-width int typecast format" << std::endl;
      abort();
    }
  } else if (from_width < to_width) {
    return convert_sign_ext(a, s, from_width, (to_width - from_width));
  } else if (from_width > to_width) {
    return mk_extract(a, to_width - 1, 0, s);
  }

  std::cerr << "Malformed cast from signedbv/fixedbv" << std::endl;
  abort();
}

smt_astt
smt_convt::convert_typecast_to_ints_from_unsigned(const typecast2t &cast)
{
  assert(!int_encoding);
  unsigned to_width = cast.type->get_width();
  smt_sortt s = convert_sort(cast.type);
  smt_astt a = convert_ast(cast.from);

  unsigned from_width = cast.from->type->get_width();

  if (from_width == to_width) {
    return a;   // output = output
  } else if (from_width < to_width) {
    return convert_zero_ext(a, s, (to_width - from_width));
  } else {
    assert(from_width > to_width);
    return mk_extract(a, to_width - 1, 0, s);
  }
}


smt_astt
smt_convt::convert_typecast_to_ints_from_bool(const typecast2t &cast)
{
  assert(!int_encoding);
  smt_sortt s = convert_sort(cast.type);
  smt_astt a = convert_ast(cast.from);

  smt_astt zero, one;
  unsigned width = cast.type->get_width();

  zero = mk_smt_bvint(BigInt(0), false, width);
  one = mk_smt_bvint(BigInt(1), false, width);

  return mk_func_app(s, SMT_FUNC_ITE, a, one, zero);
}

smt_astt
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
  type2tc int_type = machine_ptr;
  smt_sortt int_sort = convert_sort(int_type);
  typecast2tc cast_to_unsigned(int_type, cast.from);
  smt_astt target = convert_ast(cast_to_unsigned);

  // Construct array for all possible object outcomes
  std::vector<smt_astt> is_in_range;
  std::vector<smt_astt> obj_ids;
  std::vector<smt_astt> obj_starts;
  is_in_range.resize(addr_space_data.back().size());
  obj_ids.resize(addr_space_data.back().size());
  obj_starts.resize(addr_space_data.back().size());

  smt_func_kind gek = (int_encoding) ? SMT_FUNC_GTE : SMT_FUNC_BVUGTE;
  smt_func_kind lek = (int_encoding) ? SMT_FUNC_LTE : SMT_FUNC_BVULTE;

  std::map<unsigned, unsigned>::const_iterator it;
  unsigned int i;
  for (it = addr_space_data.back().begin(), i = 0;
       it != addr_space_data.back().end(); it++, i++)
  {
    unsigned id = it->first;
    obj_ids[i] = convert_terminal(constant_int2tc(int_type, BigInt(id)));

    std::stringstream ss1, ss2;
    ss1 << "__ESBMC_ptr_obj_start_" << id;
    smt_astt ptr_start = mk_smt_symbol(ss1.str(), int_sort);
    ss2 << "__ESBMC_ptr_obj_end_" << id;
    smt_astt ptr_end = mk_smt_symbol(ss2.str(), int_sort);

    obj_starts[i] = ptr_start;

    smt_astt ge = mk_func_app(boolean_sort, gek, target, ptr_start);
    smt_astt le = mk_func_app(boolean_sort, lek, target, ptr_end);
    smt_astt theand = mk_func_app(boolean_sort, SMT_FUNC_AND, ge, le);
    is_in_range[i] = theand;
  }

  // Create a fresh new variable; encode implications that if the integer is
  // in the relevant range, that the value is the relevant id / offset. If none
  // are matched, match the invalid pointer.
  // Technically C doesn't allow for any variable to hold an invalid pointer,
  // except through initialization.

  smt_sortt s = convert_sort(cast.type);
  smt_astt output = mk_fresh(s, "smt_convt::int_to_ptr");
  smt_astt output_obj = output->project(this, 0);
  smt_astt output_offs = output->project(this, 1);

  smt_func_kind subk = (int_encoding) ? SMT_FUNC_SUB : SMT_FUNC_BVSUB;

  ast_vec guards;
  for (i = 0; i < addr_space_data.back().size(); i++) {
    if (i == 1)
      continue; // Skip invalid, it contains everything.

    // Calculate ptr offset were it this
    smt_astt offs = mk_func_app(int_sort, subk, target, obj_starts[i]);

    smt_astt this_obj = obj_ids[i];
    smt_astt this_offs = offs;

    smt_astt obj_eq = this_obj->eq(this, output_obj);
    smt_astt offs_eq = this_offs->eq(this, output_offs);
    smt_astt is_eq = mk_func_app(boolean_sort, SMT_FUNC_AND, obj_eq, offs_eq);

    smt_astt in_range = is_in_range[i];
    guards.push_back(in_range);
    smt_astt imp = mk_func_app(boolean_sort, SMT_FUNC_IMPLIES, in_range, is_eq);
    assert_ast(imp);
  }

  // If none of the above, match invalid.
  smt_astt was_matched = make_disjunct(guards);
  smt_astt not_matched = mk_func_app(boolean_sort, SMT_FUNC_NOT, was_matched);

  smt_astt id =
    convert_terminal(
        constant_int2tc(int_type, pointer_logic.back().get_invalid_object()));

  smt_astt one = convert_terminal(one_ulong);
  smt_astt offs = mk_func_app(int_sort, subk, target, one);
  smt_astt inv_obj = id;
  smt_astt inv_offs = offs;

  smt_astt obj_eq = inv_obj->eq(this, output_obj);
  smt_astt offs_eq = inv_offs->eq(this, output_offs);
  smt_astt is_inv = mk_func_app(boolean_sort, SMT_FUNC_AND, obj_eq, offs_eq);

  smt_astt imp =
    mk_func_app(boolean_sort, SMT_FUNC_IMPLIES, not_matched, is_inv);
  assert_ast(imp);

  return output;
}

smt_astt
smt_convt::convert_typecast_from_ptr(const typecast2t &cast)
{

  type2tc int_type = machine_ptr;

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

smt_astt
smt_convt::convert_typecast_to_struct(const typecast2t &cast)
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

  smt_sortt fresh_sort = convert_sort(cast.type);
  smt_astt fresh = tuple_api->tuple_fresh(fresh_sort);
  smt_astt src_ast = convert_ast(cast.from);

  if (same_format) {
    // Alas, Z3 considers field names as being part of the type, so we can't
    // just consider the source expression to be the casted expression.
    i2 = 0;
    forall_types(it, struct_type_to.members) {
      smt_astt args[2];
      args[0] = src_ast->project(this, i2);
      args[1] = fresh->project(this, i2);
      assert_ast(args[0]->eq(this, args[1]));
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

      smt_astt args[2];
      args[0] = src_ast->project(this, i3);
      args[1] = fresh->project(this, i2);

      assert_ast(args[0]->eq(this, args[1]));
      i2++;
    }
  }

  return fresh;
}

smt_astt
smt_convt::convert_typecast(const expr2tc &expr)
{

  const typecast2t &cast = to_typecast2t(expr);
  if (cast.type == cast.from->type)
    return convert_ast(cast.from);

  // Casts to and from pointers need to be addressed all as one
  if (is_pointer_type(cast.type)) {
    return convert_typecast_to_ptr(cast);
  } else if (is_pointer_type(cast.from)) {
    return convert_typecast_from_ptr(cast);
  }

  // Otherwise, look at the result type.
  if (is_bool_type(cast.type)) {
    return convert_typecast_to_bool(cast);
  } else if (is_fixedbv_type(cast.type) && !int_encoding) {
    return convert_typecast_to_fixedbv_nonint(expr);
  } else if (is_bv_type(cast.type) || is_fixedbv_type(cast.type)) {
    return convert_typecast_to_ints(cast);
  } else if (is_floatbv_type(cast.type)) {
    return mk_smt_typecast_to_bvfloat(cast);
  } else if (is_struct_type(cast.type)) {
    return convert_typecast_to_struct(cast);
  } else if (is_union_type(cast.type)) {
    if (base_type_eq(cast.type, cast.from->type, ns)) {
      return convert_ast(cast.from); // No additional conversion required
    } else {
      std::cerr << "Can't typecast between unions" << std::endl;
      abort();
    }
  }

  std::cerr << "Typecast for unexpected type" << std::endl;
  expr->dump();
  abort();
}
