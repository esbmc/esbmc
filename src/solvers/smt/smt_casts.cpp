#include <solvers/smt/smt_conv.h>
#include <sstream>
#include <util/base_type.h>
#include <util/expr_util.h>

smt_astt smt_convt::convert_typecast_to_bool(const typecast2t &cast)
{
  if(is_pointer_type(cast.from))
  {
    // Convert to two casts.
    typecast2tc to_int(machine_ptr, cast.from);
    equality2tc as_bool(gen_zero(machine_ptr), to_int);
    return convert_ast(as_bool);
  }

  notequal2tc neq(cast.from, gen_zero(cast.from->type));
  return convert_ast(neq);
}

smt_astt smt_convt::convert_typecast_to_fixedbv_nonint(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);

  if(is_pointer_type(cast.from))
  {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  if(is_bv_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_bv(expr);
  if(is_bool_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_bool(expr);
  else if(is_fixedbv_type(cast.from))
    return convert_typecast_to_fixedbv_nonint_from_fixedbv(expr);

  std::cerr << "unexpected typecast to fixedbv" << std::endl;
  abort();
}

smt_astt smt_convt::convert_typecast_to_fixedbv_nonint_from_bv(
  const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;
  assert(is_bv_type(cast.from));

  smt_astt a = convert_ast(cast.from);

  unsigned from_width = cast.from->type->get_width();

  smt_astt frontpart;
  if(from_width == to_integer_bits)
  {
    // Just concat fraction ozeros at the bottom
    frontpart = a;
  }
  else if(from_width > to_integer_bits)
  {
    frontpart = mk_extract(a, to_integer_bits - 1, 0);
  }
  else
  {
    assert(from_width < to_integer_bits);
    frontpart = mk_sign_ext(a, to_integer_bits - from_width);
  }

  // Make all zeros fraction bits
  smt_astt zero_fracbits = mk_smt_bv(BigInt(0), to_fraction_bits);
  return mk_concat(frontpart, zero_fracbits);
}

smt_astt smt_convt::convert_typecast_to_fixedbv_nonint_from_bool(
  const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_integer_bits = fbvt.integer_bits;
  assert(is_bool_type(cast.from));

  smt_astt a = convert_ast(cast.from);

  smt_astt zero = mk_smt_bv(BigInt(0), to_integer_bits);
  smt_astt one = mk_smt_bv(BigInt(1), to_integer_bits);
  smt_astt switched = mk_ite(a, zero, one);
  return mk_concat(switched, zero);
}

smt_astt smt_convt::convert_typecast_to_fixedbv_nonint_from_fixedbv(
  const expr2tc &expr)
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

  // FIXME: conversion here for to_int_bits > from_int_bits is factually
  // broken, run 01_cbmc_Fixedbv8 with --no-simplify

  // The plan here is to extract the magnitude and fraction from the source
  // fbv, extend or truncate them appropriately, then concatenate them.

  // Start with the magnitude
  if(to_integer_bits <= from_integer_bits)
  {
    magnitude = mk_extract(
      a, (from_fraction_bits + to_integer_bits - 1), from_fraction_bits);
  }
  else
  {
    assert(to_integer_bits > from_integer_bits);
    smt_astt ext = mk_extract(a, from_width - 1, from_fraction_bits);

    unsigned int additional_bits = to_integer_bits - from_integer_bits;
    magnitude = mk_sign_ext(ext, additional_bits);
  }

  // Followed by the fraction part
  if(to_fraction_bits <= from_fraction_bits)
  {
    fraction = mk_extract(
      a, from_fraction_bits - 1, from_fraction_bits - to_fraction_bits);
  }
  else
  {
    assert(to_fraction_bits > from_fraction_bits);

    // Increase the size of the fraction by adding zeros on the end. This is
    // not a zero extension because they're at the end, not the start
    smt_astt src_fraction = mk_extract(a, from_fraction_bits - 1, 0);
    smt_astt zeros =
      mk_smt_bv(BigInt(0), to_fraction_bits - from_fraction_bits);

    fraction = mk_concat(src_fraction, zeros);
  }

  // Finally, concatenate the adjusted magnitude / fraction
  return mk_concat(magnitude, fraction);
}

smt_astt smt_convt::convert_typecast_to_fpbv(const typecast2t &cast)
{
  // Convert each type
  if(is_bool_type(cast.from))
  {
    // For bools, there is no direct conversion, so the cast is
    // transformed into fpa = b ? 1 : 0;
    return mk_ite(
      convert_ast(cast.from),
      convert_ast(gen_one(cast.type)),
      convert_ast(gen_zero(cast.type)));
  }

  if(is_unsignedbv_type(cast.from))
    return fp_api->mk_smt_typecast_ubv_to_fpbv(
      convert_ast(cast.from),
      convert_sort(cast.type),
      convert_rounding_mode(cast.rounding_mode));

  if(is_signedbv_type(cast.from))
    return fp_api->mk_smt_typecast_sbv_to_fpbv(
      convert_ast(cast.from),
      convert_sort(cast.type),
      convert_rounding_mode(cast.rounding_mode));

  if(is_floatbv_type(cast.from))
    return fp_api->mk_smt_typecast_from_fpbv_to_fpbv(
      convert_ast(cast.from),
      convert_sort(cast.type),
      convert_rounding_mode(cast.rounding_mode));

  std::cerr << "Unexpected type in typecast to fpbv\n";
  abort();
}

smt_astt smt_convt::convert_typecast_from_fpbv(const typecast2t &cast)
{
  if(is_unsignedbv_type(cast.type))
    return fp_api->mk_smt_typecast_from_fpbv_to_ubv(
      convert_ast(cast.from), cast.type->get_width());

  if(is_signedbv_type(cast.type))
    return fp_api->mk_smt_typecast_from_fpbv_to_sbv(
      convert_ast(cast.from), cast.type->get_width());

  if(is_floatbv_type(cast.type))
    return fp_api->mk_smt_typecast_from_fpbv_to_fpbv(
      convert_ast(cast.from),
      convert_sort(cast.type),
      convert_rounding_mode(cast.rounding_mode));

  std::cerr << "Unexpected type in typecast from fpbv\n";
  abort();
}

smt_astt smt_convt::convert_typecast_to_ints(const typecast2t &cast)
{
  if(int_encoding)
    return convert_typecast_to_ints_intmode(cast);

  if(is_signedbv_type(cast.from) || is_fixedbv_type(cast.from))
    return convert_typecast_to_ints_from_fbv_sint(cast);

  if(is_unsignedbv_type(cast.from))
    return convert_typecast_to_ints_from_unsigned(cast);
  if(is_floatbv_type(cast.from))
    return convert_typecast_from_fpbv(cast);

  if(is_bool_type(cast.from))
    return convert_typecast_to_ints_from_bool(cast);

  std::cerr << "Unexpected type in int/ptr typecast" << std::endl;
  abort();
}

smt_astt smt_convt::convert_typecast_to_ints_intmode(const typecast2t &cast)
{
  assert(int_encoding);
  // Integer-mode conversion of integers. Immediately, we don't care about
  // bit widths, to the extent that any fixedbv <=> fixedbv conversion can
  // remain a real, and any {un,}signedbv <=> {un,}signedbv conversion can
  // remain an int. The only thing we actually need to care about is the
  // conversion between ints and reals.

  if(is_fixedbv_type(cast.type) && is_fixedbv_type(cast.from))
    return convert_ast(cast.from);

  if(is_bv_type(cast.type) && is_bv_type(cast.from))
    // NB: this means negative numbers assigned to unsigned ints remain
    // negative. This IMO is one of the inaccuracies accepted by the use of
    // ir mode.
    return convert_ast(cast.from);

  smt_astt a = convert_ast(cast.from);

  // Handle conversions from booleans
  if(is_bool_type(cast.from))
  {
    smt_astt zero, one;
    if(is_bv_type(cast.type))
    {
      zero = mk_smt_int(BigInt(0), false);
      one = mk_smt_int(BigInt(1), false);
    }
    else
    {
      zero = mk_smt_real("0");
      one = mk_smt_real("1");
    }

    return mk_ite(a, one, zero);
  }

  // Otherwise, we're looking at a cast between reals and int sorts.
  if(is_fixedbv_type(cast.type))
  {
    assert(is_bv_type(cast.from));
    return mk_int2real(a);
  }

  assert(is_bv_type(cast.type));
  assert(is_fixedbv_type(cast.from));
  return round_real_to_int(a);
}

smt_astt smt_convt::convert_typecast_to_ints_from_fbv_sint(
  const typecast2t &cast)
{
  assert(!int_encoding);
  unsigned to_width = cast.type->get_width();
  smt_astt a = convert_ast(cast.from);

  unsigned from_width = cast.from->type->get_width();

  if(from_width == to_width)
  {
    if(is_fixedbv_type(cast.from) && is_bv_type(cast.type))
      return round_fixedbv_to_int(a, from_width, to_width);

    if(
      (is_signedbv_type(cast.type) && is_unsignedbv_type(cast.from)) ||
      (is_unsignedbv_type(cast.type) && is_signedbv_type(cast.from)))
      // Operands have differing signs (and same width). Just return.
      return convert_ast(cast.from);

    std::cerr << "Unrecognized equal-width int typecast format" << std::endl;
    abort();
  }

  if(from_width < to_width)
    return mk_sign_ext(a, to_width - from_width);

  if(from_width > to_width)
    return mk_extract(a, to_width - 1, 0);

  std::cerr << "Malformed cast from signedbv/fixedbv" << std::endl;
  abort();
}

smt_astt smt_convt::convert_typecast_to_ints_from_unsigned(
  const typecast2t &cast)
{
  assert(!int_encoding);
  unsigned to_width = cast.type->get_width();

  smt_astt a = convert_ast(cast.from);

  unsigned from_width = cast.from->type->get_width();

  if(from_width == to_width)
    return a; // output = output

  if(from_width < to_width)
    return mk_zero_ext(a, (to_width - from_width));

  assert(from_width > to_width);
  return mk_extract(a, to_width - 1, 0);
}

smt_astt smt_convt::convert_typecast_to_ints_from_bool(const typecast2t &cast)
{
  assert(!int_encoding);
  smt_astt a = convert_ast(cast.from);

  smt_astt zero, one;
  unsigned width = cast.type->get_width();

  zero = mk_smt_bv(BigInt(0), width);
  one = mk_smt_bv(BigInt(1), width);

  return mk_ite(a, one, zero);
}

smt_astt smt_convt::convert_typecast_to_ptr(const typecast2t &cast)
{
  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if(is_pointer_type(cast.from))
    return convert_ast(cast.from);

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

  std::map<unsigned, unsigned>::const_iterator it;
  unsigned int i;
  for(it = addr_space_data.back().begin(), i = 0;
      it != addr_space_data.back().end();
      it++, i++)
  {
    unsigned id = it->first;
    obj_ids[i] = convert_terminal(constant_int2tc(int_type, BigInt(id)));

    std::stringstream ss1, ss2;
    ss1 << "__ESBMC_ptr_obj_start_" << id;
    smt_astt ptr_start = mk_smt_symbol(ss1.str(), int_sort);
    ss2 << "__ESBMC_ptr_obj_end_" << id;
    smt_astt ptr_end = mk_smt_symbol(ss2.str(), int_sort);

    obj_starts[i] = ptr_start;

    smt_astt ge =
      int_encoding ? mk_ge(target, ptr_start) : mk_bvuge(target, ptr_start);
    smt_astt le =
      int_encoding ? mk_le(target, ptr_end) : mk_bvule(target, ptr_end);
    smt_astt theand = mk_and(ge, le);
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

  ast_vec guards;
  for(i = 0; i < addr_space_data.back().size(); i++)
  {
    if(i == 1)
      continue; // Skip invalid, it contains everything.

    // Calculate ptr offset were it this
    smt_astt offs = int_encoding ? mk_sub(target, obj_starts[i])
                                 : mk_bvsub(target, obj_starts[i]);

    smt_astt this_obj = obj_ids[i];
    smt_astt this_offs = offs;

    smt_astt obj_eq = this_obj->eq(this, output_obj);
    smt_astt offs_eq = this_offs->eq(this, output_offs);
    smt_astt is_eq = mk_and(obj_eq, offs_eq);

    smt_astt in_range = is_in_range[i];
    guards.push_back(in_range);
    smt_astt imp = mk_implies(in_range, is_eq);
    assert_ast(imp);
  }

  // If none of the above, match invalid.
  smt_astt was_matched = make_n_ary(this, &smt_convt::mk_or, guards);
  smt_astt not_matched = mk_not(was_matched);

  smt_astt id = convert_terminal(
    constant_int2tc(int_type, pointer_logic.back().get_invalid_object()));

  smt_astt one = convert_terminal(gen_ulong(1));
  smt_astt offs = int_encoding ? mk_sub(target, one) : mk_bvsub(target, one);
  smt_astt inv_obj = id;
  smt_astt inv_offs = offs;

  smt_astt obj_eq = inv_obj->eq(this, output_obj);
  smt_astt offs_eq = inv_offs->eq(this, output_offs);
  smt_astt is_inv = mk_and(obj_eq, offs_eq);

  smt_astt imp = mk_implies(not_matched, is_inv);
  assert_ast(imp);

  return output;
}

smt_astt smt_convt::convert_typecast_from_ptr(const typecast2t &cast)
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

smt_astt smt_convt::convert_typecast_to_struct(const typecast2t &cast)
{
  const struct_type2t &struct_type_from = to_struct_type(cast.from->type);
  const struct_type2t &struct_type_to = to_struct_type(cast.type);

  std::vector<type2tc> new_members;
  std::vector<irep_idt> new_names;
  new_members.reserve(struct_type_to.members.size());
  new_names.reserve(struct_type_to.members.size());

  // This all goes to pot when we consider polymorphism, and in particular,
  // multiple inheritance. So, for normal structs, as usual check that each
  // field has a compatible type. But for classes, check that either they're
  // the same class, or the source is a subclass of the target type. If so,
  // we just select out the common fields, which drops any additional data in
  // the subclass.

  unsigned int i = 0;
  bool same_format = true;
  if(is_subclass_of(cast.from->type, cast.type, ns))
  {
    same_format = false; // then we're fine
  }
  else if(struct_type_from.name == struct_type_to.name)
  {
    ; // Also fine
  }
  else
  {
    // Check that these two different structs have the same format.
    for(auto const &it : struct_type_to.members)
    {
      if(!base_type_eq(struct_type_from.members[i], it, ns))
      {
        std::cerr << "Incompatible struct in cast-to-struct" << std::endl;
        abort();
      }

      i++;
    }
  }

  smt_sortt fresh_sort = convert_sort(cast.type);
  smt_astt fresh = tuple_api->tuple_fresh(fresh_sort);
  smt_astt src_ast = convert_ast(cast.from);

  unsigned int i2 = 0;
  if(same_format)
  {
    // Alas, Z3 considers field names as being part of the type, so we can't
    // just consider the source expression to be the casted expression.
    for(; i2 < struct_type_to.members.size(); i2++)
    {
      smt_astt args[2];
      args[0] = src_ast->project(this, i2);
      args[1] = fresh->project(this, i2);
      assert_ast(args[0]->eq(this, args[1]));
    }
  }
  else
  {
    // Due to inheritance, these structs don't have the same format. Therefore
    // we have to look up source fields by matching the field names between
    // structs, then using their index numbers construct equalities between
    // fields in the source value and a fresh value.
    for(auto const &it : struct_type_to.member_names)
    {
      // Linear search, yay :(
      unsigned int i3 = 0;
      for(auto const &it2 : struct_type_from.member_names)
      {
        if(it == it2)
          break;
        i3++;
      }

      assert(
        i3 != struct_type_from.member_names.size() &&
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

smt_astt smt_convt::convert_typecast(const expr2tc &expr)
{
  const typecast2t &cast = to_typecast2t(expr);
  if(cast.type == cast.from->type)
    return convert_ast(cast.from);

  // Casts to and from pointers need to be addressed all as one
  if(is_pointer_type(cast.type))
    return convert_typecast_to_ptr(cast);

  if(is_pointer_type(cast.from))
    return convert_typecast_from_ptr(cast);

  // Otherwise, look at the result type.
  if(is_bool_type(cast.type))
    return convert_typecast_to_bool(cast);

  if(is_fixedbv_type(cast.type) && !int_encoding)
    return convert_typecast_to_fixedbv_nonint(expr);

  if(is_bv_type(cast.type) || is_fixedbv_type(cast.type))
    return convert_typecast_to_ints(cast);

  if(is_floatbv_type(cast.type))
    return convert_typecast_to_fpbv(cast);

  if(is_struct_type(cast.type))
    return convert_typecast_to_struct(cast);

  if(is_union_type(cast.type))
  {
    if(base_type_eq(cast.type, cast.from->type, ns))
      return convert_ast(cast.from); // No additional conversion required

    std::cerr << "Can't typecast between unions" << std::endl;
    abort();
  }

  std::cerr << "Typecast for unexpected type" << std::endl;
  expr->dump();
  abort();
}
