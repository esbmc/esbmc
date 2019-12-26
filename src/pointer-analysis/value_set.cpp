/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <langapi/language_util.h>
#include <pointer-analysis/value_set.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/context.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/simplify_expr.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

object_numberingt value_sett::object_numbering;
object_number_numberingt value_sett::obj_numbering_refset;

void value_sett::output(std::ostream &out) const
{
  // Iterate over all tracked variables, dumping a list of all the things it
  // might point at.
  for(const auto &value : values)
  {
    std::string identifier, display_name;

    const entryt &e = value.second;

    if(has_prefix(e.identifier, "value_set::dynamic_object"))
    {
      display_name = e.identifier + e.suffix;
      identifier = "";
    }
    else if(e.identifier == "value_set::return_value")
    {
      display_name = "RETURN_VALUE" + e.suffix;
      identifier = "";
    }
    else
    {
#if 0
      const symbolt &symbol=ns.lookup(e.identifier);
      display_name=symbol.display_name()+e.suffix;
      identifier=symbol.name;
#else
      identifier = e.identifier;
      display_name = identifier + e.suffix;
#endif
    }

    out << display_name;

    out << " = { ";

    unsigned width = 0;

    for(object_mapt::const_iterator o_it = e.object_map.begin();
        o_it != e.object_map.end();
        o_it++)
    {
      const expr2tc &o = object_numbering[o_it->first];

      std::string result;

      // Display invalid / unknown objects as just that,
      if(is_invalid2t(o) || is_unknown2t(o))
        result = from_expr(ns, identifier, o);
      else
      {
        // Everything else, display as a triple of <object, offset, type>.
        result = "<" + from_expr(ns, identifier, o) + ", ";

        if(o_it->second.offset_is_set)
          result += integer2string(o_it->second.offset) + "";
        else
          result += "*";

        result += ", " + from_type(ns, identifier, o->type);

        result += ">";
      }

      // Actually print.
      out << result;

      width += result.size();

      object_mapt::const_iterator next(o_it);
      next++;

      if(next != e.object_map.end())
      {
        out << ", ";
        if(width >= 40)
          out << "\n      ";
      }
    }

    out << " } " << std::endl;
  }
}

expr2tc value_sett::to_expr(object_mapt::const_iterator it) const
{
  const expr2tc &object = object_numbering[it->first];

  if(is_invalid2t(object) || is_unknown2t(object))
    return object;

  expr2tc offs;
  if(it->second.offset_is_set)
    offs = constant_int2tc(index_type2(), it->second.offset);
  else
    offs = unknown2tc(index_type2());

  expr2tc obj = object_descriptor2tc(
    object->type, object, offs, it->second.offset_alignment);
  return obj;
}

bool value_sett::make_union(const value_sett::valuest &new_values, bool keepnew)
{
  bool result = false;

  // Iterate over all new values; if they're in the current value set, merge
  // them. If not, only merge it in if keepnew is true.
  for(const auto &new_value : new_values)
  {
    valuest::iterator it2 = values.find(new_value.first);

    // If the new variable isnt in this' set,
    if(it2 == values.end())
    {
      // We always track these when merging value sets, as these store data
      // that's transfered back and forth between function calls. So, the
      // variables not existing in the state we're merging into is irrelevant.
      if(
        has_prefix(
          id2string(new_value.second.identifier),
          "value_set::dynamic_object") ||
        new_value.second.identifier == "value_set::return_value" || keepnew)
      {
        values.insert(new_value);
        result = true;
      }

      continue;
    }

    // The variable was in this' set, merge the values.
    entryt &e = it2->second;
    const entryt &new_e = new_value.second;

    if(make_union(e.object_map, new_e.object_map))
      result = true;
  }

  return result;
}

bool value_sett::make_union(object_mapt &dest, const object_mapt &src) const
{
  bool result = false;

  // Merge the pointed at objects in src into dest.
  for(object_mapt::const_iterator it = src.begin(); it != src.end(); it++)
  {
    if(insert(dest, it))
      result = true;
  }

  return result;
}

void value_sett::get_value_set(const expr2tc &expr, value_setst::valuest &dest)
  const
{
  object_mapt object_map;
  // Fetch all values into object_map,
  get_value_set(expr, object_map);

  // Convert values into expressions to return.
  for(object_mapt::const_iterator it = object_map.begin();
      it != object_map.end();
      it++)
    dest.push_back(to_expr(it));
}

void value_sett::get_value_set(const expr2tc &expr, object_mapt &dest) const
{
  // Simplify expr if possible,
  expr2tc new_expr = expr;
  simplify(new_expr);

  // Then, start fetching values.
  get_value_set_rec(new_expr, dest, "", new_expr->type);
}

void value_sett::get_value_set_rec(
  const expr2tc &expr,
  object_mapt &dest,
  const std::string &suffix,
  const type2tc &original_type) const
{
  if(is_unknown2t(expr) || is_invalid2t(expr))
  {
    // Unknown / invalid exprs mean we just point at something unknown (and
    // potentially invalid).
    insert(dest, unknown2tc(original_type), mp_integer(0));
    return;
  }
  if(is_index2t(expr))
  {
    // This is an index, fetch values from the array being indexed.
    const index2t &idx = to_index2t(expr);

#ifndef NDEBUG
    const type2tc &source_type = idx.source_value->type;
    assert(is_array_type(source_type) || is_string_type(source_type));
#endif

    // Attach '[]' to the suffix, identifying the variable tracking all the
    // pointers in this array.
    get_value_set_rec(idx.source_value, dest, "[]" + suffix, original_type);

    return;
  }
  else if(is_member2t(expr))
  {
    // We're selecting a member variable of a structure: fetch the values it
    // might point at.
    const member2t &memb = to_member2t(expr);

#ifndef NDEBUG
    const type2tc &source_type = memb.source_value->type;
    assert(is_struct_type(source_type) || is_union_type(source_type));
#endif

    // Add '.$field' to the suffix, identifying the member from the other
    // members of the struct's variable.
    get_value_set_rec(
      memb.source_value,
      dest,
      "." + memb.member.as_string() + suffix,
      original_type);
    return;
  }
  else if(is_symbol2t(expr))
  {
    // This is a symbol, and if it's a pointer then this expression might
    // evalutate to what it points at. So, return this symbols value set.
    const symbol2t &sym = to_symbol2t(expr);

    // If it's null however, create a null_object2t with the appropriate type.
    if(sym.thename == "NULL" && is_pointer_type(expr))
    {
      const pointer_type2t &ptr_ref = to_pointer_type(expr->type);
      typet subtype = migrate_type_back(ptr_ref.subtype);
      if(subtype.id() == "symbol")
        subtype = ns.follow(subtype);

      expr2tc tmp = null_object2tc(ptr_ref.subtype);
      insert(dest, tmp, mp_integer(0));
      return;
    }

    // Look up this symbol, with the given suffix to distinguish any arrays or
    // members we've picked out of it at a higher level.
    valuest::const_iterator v_it =
      values.find(string_wrapper(sym.get_symbol_name() + suffix));

    // If it points at things, put those things into the destination object map.
    if(v_it != values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }
  else if(is_if2t(expr))
  {
    // This expression might evaluate to either side of this if (assuming that
    // the simplifier couldn't simplify it away. Grab the value set from either
    // side.
    const if2t &ifval = to_if2t(expr);

    get_value_set_rec(ifval.true_value, dest, suffix, original_type);
    get_value_set_rec(ifval.false_value, dest, suffix, original_type);

    return;
  }
  else if(is_address_of2t(expr))
  {
    // The set of things this expression might point at is the set of things
    // that might be the operand to this address-of. So, get the reference set
    // of things it refers to, rather than the value set (of things it points
    // to).
    const address_of2t &addrof = to_address_of2t(expr);
    get_reference_set(addrof.ptr_obj, dest);
    return;
  }
  else if(is_dereference2t(expr))
  {
    // Fetch the set of things that this dereference might point at... That
    // means if we have the code:
    //   int *a = NULL;
    //   int **b = &a;
    //   *b;
    // Then we're evaluating the final line, what does *b point at? To do this,
    // take all the things that (*b) refers to, which performs the actual
    // dereference itself. We then have a list of things that b might point at
    // (in this case just a); so we then compute the value set of all those
    // things.
    object_mapt reference_set;
    // Get reference set of dereference; this evaluates the dereference itself.
    get_reference_set(expr, reference_set);

    // Then get the value set of all the pointers we might dereference to.
    for(const auto &it1 : reference_set)
    {
      const expr2tc &object = object_numbering[it1.first];
      get_value_set_rec(object, dest, suffix, original_type);
    }

    return;
  }
  else if(is_constant_expr(expr))
  {
    // Constant numbers aren't pointers. Null check is in the value set code
    // for symbols.
    return;
  }
  else if(is_typecast2t(expr))
  {
    // Push straight through typecasts.
    const typecast2t &cast = to_typecast2t(expr);
    get_value_set_rec(cast.from, dest, suffix, original_type);
    return;
  }
  else if(is_bitcast2t(expr))
  {
    // Bitcasts are just typecasts with additional semantics
    const bitcast2t &cast = to_bitcast2t(expr);
    get_value_set_rec(cast.from, dest, suffix, original_type);
    return;
  }
  else if(is_add2t(expr) || is_sub2t(expr))
  {
    // Consider pointer arithmetic. This takes takes the form of finding the
    // value sets of the operands, then speculating on how the addition /
    // subtraction affects the offset.
    if(is_pointer_type(expr))
    {
      // find the pointer operand
      // XXXjmorse - polymorphism.
      const expr2tc &op0 =
        (is_add2t(expr)) ? to_add2t(expr).side_1 : to_sub2t(expr).side_1;
      const expr2tc &op1 =
        (is_add2t(expr)) ? to_add2t(expr).side_2 : to_sub2t(expr).side_2;

      assert(
        !(is_pointer_type(op0) && is_pointer_type(op1)) &&
        "Cannot have pointer arithmetic with two pointers as operands");

      const expr2tc &ptr_op = (is_pointer_type(op0)) ? op0 : op1;
      const expr2tc &non_ptr_op = (is_pointer_type(op0)) ? op1 : op0;

      // Find out what the pointer operand points at, and suck that data into
      // a new object map.
      object_mapt pointer_expr_set;
      get_value_set_rec(ptr_op, pointer_expr_set, "", ptr_op->type);

      // Calculate the offset caused by this addition, in _bytes_. Involves
      // pointer arithmetic. We also use the _perceived_ type of what we're
      // adding or subtracting from/to, it might be being typecasted.
      const type2tc &subtype = to_pointer_type(ptr_op->type).subtype;
      mp_integer total_offs(0);
      bool is_const = false;
      try
      {
        if(is_constant_int2t(non_ptr_op))
        {
          if(to_constant_int2t(non_ptr_op).value.is_zero())
          {
            total_offs = 0;
          }
          else
          {
            if(is_empty_type(subtype))
              throw new type2t::symbolic_type_excp();

            // Potentially rename,
            const type2tc renamed = ns.follow(subtype);
            mp_integer elem_size = type_byte_size(renamed);
            const mp_integer &val = to_constant_int2t(non_ptr_op).value;
            total_offs = val * elem_size;
            if(is_sub2t(expr))
              total_offs.negate();
          }
          is_const = true;
        }
        else
        {
          is_const = false;
        }
      }
      catch(array_type2t::dyn_sized_array_excp *e)
      { // Nondet'ly sized.
      }
      catch(array_type2t::inf_sized_array_excp *e)
      {
      }
      catch(type2t::symbolic_type_excp *e)
      {
        // This vastly annoying piece of code is making operations on void
        // pointers, or worse. If a void pointer, treat the multiplier of the
        // addition as being one. If not void pointer, throw cookies.
        if(is_empty_type(subtype))
        {
          total_offs = to_constant_int2t(non_ptr_op).value;
          is_const = true;
        }
        else
        {
          std::cerr << "Pointer arithmetic on type where we can't determine ";
          std::cerr << "size:" << std::endl;
          std::cerr << subtype->pretty(0) << std::endl;
          abort();
        }
      }

      // For each object, update its offset data according to the integer
      // offset to this expr. Potential outcomes are keeping it nondet, making
      // it nondet, or calculating a new static offset.
      for(const auto &it : pointer_expr_set)
      {
        objectt object = it.second;

        unsigned int nat_align =
          get_natural_alignment(object_numbering[it.first]);
        unsigned int ptr_align = get_natural_alignment(ptr_op);

        if(is_const && object.offset_is_set)
        {
          // Both are const; we can accumulate offsets;
          object.offset += total_offs;
        }
        else if(is_const && !object.offset_is_set)
        {
          // Offset is const, but existing pointer isn't. The alignment is now
          // at least as small as the operand alignment.
          object.offset_alignment =
            std::min(nat_align, object.offset_alignment);
        }
        else if(!is_const && object.offset_is_set)
        {
          // Nondet but aligned offset from arithmetic; but offset set in
          // current object. Take the minimum alignment again.
          unsigned int offset_align = 0;
          if((object.offset % nat_align) != 0)
          {
            // We have some kind of offset into this data object, but it's less
            // than the data objects natural alignment. So, the maximum
            // alignment we can have is that of the pointer type being added
            // or subtracted. The minimum, depends on the offset into the
            // data object we're pointing at.
            offset_align = ptr_align;
            if(object.offset % ptr_align != 0)
              // To complex to calculate; clamp to bytes.
              offset_align = 1;
          }
          else
          {
            offset_align = nat_align;
          }

          object.offset_is_set = false;
          object.offset_alignment = std::min(nat_align, offset_align);
        }
        else
        {
          // Final case: nondet offset from operation, and nondet offset in
          // the current object. So, just take the minimum available.
          object.offset_alignment =
            std::min(nat_align, object.offset_alignment);
        }

        // Once updated, store object reference into destination map.
        insert(dest, it.first, object);
      }

      return;
    }
  }
  else if(is_sideeffect2t(expr))
  {
    // Consider a (potentially memory allocating) side effect. Perform crazy
    // black (and possibly broken) magic to track said memory during static
    // analysis.
    // During symbolic execution, the only assignments handed to value_sett
    // have all the sideeffects taken out of them (as they're SSA assignments),
    // so this is never triggered.
    const sideeffect2t &side = to_sideeffect2t(expr);
    switch(side.kind)
    {
    case sideeffect2t::malloc:
    {
      assert(suffix == "");
      const type2tc &dynamic_type = side.alloctype;

      expr2tc locnum = gen_ulong(location_number);
      dynamic_object2tc dynobj(dynamic_type, locnum, false, false);

      insert(dest, dynobj, mp_integer(0));
    }
      return;

    case sideeffect2t::cpp_new:
    case sideeffect2t::cpp_new_arr:
    {
      assert(suffix == "");
      assert(is_pointer_type(side.type));

      expr2tc locnum = gen_ulong(location_number);

      const pointer_type2t &ptr = to_pointer_type(side.type);

      dynamic_object2tc dynobj(ptr.subtype, locnum, false, false);

      insert(dest, dynobj, mp_integer(0));
    }
      return;
    case sideeffect2t::nondet:
      // Introduction of nondeterminism does not introduce new pointer vars
      return;
    default:
      std::cerr << "Unexpected side-effect: " << expr->pretty(0) << std::endl;
      abort();
    }
  }
  else if(is_constant_struct2t(expr))
  {
    // The use of an explicit constant struct value evaluates to it's address.
    address_of2tc tmp(expr->type, expr);
    insert(dest, tmp, mp_integer(0));
    return;
  }
  else if(is_with2t(expr))
  {
    // Consider an array/struct update: the pointer we evaluate to may be in
    // the base array/struct, or depending on the index may be the update value.
    // So, consider both.
    // XXX jmorse -- this could be improved. What if source_value is a constant
    // array or something?
    const with2t &with = to_with2t(expr);

    // this is the array/struct
    object_mapt tmp_map0;
    get_value_set_rec(with.source_value, tmp_map0, suffix, original_type);

    // this is the update value -- note NO SUFFIX
    object_mapt tmp_map2;
    get_value_set_rec(with.update_value, tmp_map2, "", original_type);

    make_union(dest, tmp_map0);
    make_union(dest, tmp_map2);
  }
  else if(is_constant_array_of2t(expr) || is_constant_array2t(expr))
  {
    // these are supposed to be done by assign()
    assert(0 && "Encountered array irep in get_value_set_rec");
  }
  else if(is_dynamic_object2t(expr))
  {
    const dynamic_object2t &dyn = to_dynamic_object2t(expr);

    assert(is_constant_int2t(dyn.instance));
    const constant_int2t &intref = to_constant_int2t(dyn.instance);
    std::string idnum = integer2string(intref.value);
    const std::string name = "value_set::dynamic_object" + idnum + suffix;

    // look it up
    valuest::const_iterator v_it = values.find(string_wrapper(name));

    if(v_it != values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }
  else if(is_concat2t(expr))
  {
    get_byte_stitching_value_set(expr, dest, suffix, original_type);
    return;
  }
  else if(is_byte_extract2t(expr))
  {
    // This is cropping up when one assigns, for example, a pointer into a
    // byte array. The lhs gets portions of the pointer, bitcasted and then
    // byte extracted on the lhs. Thus, we need to blast through the byte
    // extract.
    const byte_extract2t &be = to_byte_extract2t(expr);
    get_value_set_rec(be.source_value, dest, suffix, original_type);
    return;
  }

  // If none of those expressions matched, then we don't really know what this
  // expression evaluates to. So just record it as being unknown.
  unknown2tc tmp(original_type);
  insert(dest, tmp, mp_integer(0));
}

void value_sett::get_byte_stitching_value_set(
  const expr2tc &expr,
  object_mapt &dest,
  const std::string &suffix,
  const type2tc &original_type) const
{
  if(is_concat2t(expr))
  {
    const concat2t &ref = to_concat2t(expr);

    get_byte_stitching_value_set(ref.side_1, dest, suffix, original_type);
    get_byte_stitching_value_set(ref.side_2, dest, suffix, original_type);
  }
  else if(is_lshr2t(expr))
  {
    const lshr2t &ref = to_lshr2t(expr);

    get_byte_stitching_value_set(ref.side_1, dest, suffix, original_type);
  }
  else if(is_byte_extract2t(expr))
  {
    const byte_extract2t &ref = to_byte_extract2t(expr);
    // XXX XXX XXX this knackers offsets
    get_value_set_rec(ref.source_value, dest, suffix, original_type);
  }
  else
  {
    get_value_set_rec(expr, dest, suffix, original_type);
  }
}

void value_sett::get_reference_set(
  const expr2tc &expr,
  value_setst::valuest &dest) const
{
  // Fetch all the symbols expr refers to into this object map.
  object_mapt object_map;
  get_reference_set(expr, object_map);

  // Then convert to expressions into the destination list.
  for(object_mapt::const_iterator it = object_map.begin();
      it != object_map.end();
      it++)
    dest.push_back(to_expr(it));
}

void value_sett::get_reference_set_rec(const expr2tc &expr, object_mapt &dest)
  const
{
  if(
    is_symbol2t(expr) || is_dynamic_object2t(expr) ||
    is_constant_string2t(expr) || is_constant_array2t(expr))
  {
    // Any symbol we refer to, store into the destination object map.
    // Given that this is a simple symbol, we can be sure that the offset to
    // it is zero.
    insert(dest, expr, objectt(true, 0));
    return;
  }
  if(is_dereference2t(expr))
  {
    // The set of variables referred to here are the set of things the operand
    // may point at. So, find its value set, and return that.
    const dereference2t &deref = to_dereference2t(expr);
    get_value_set_rec(deref.value, dest, "", deref.type);
    return;
  }
  else if(is_index2t(expr))
  {
    // This index may be dereferencing a pointer. So, get the reference set of
    // the source value, and store a reference to all those things.
    const index2t &index = to_index2t(expr);

    assert(
      is_array_type(index.source_value) || is_string_type(index.source_value));

    // Compute the offset introduced by this index.
    mp_integer index_offset;
    bool has_const_index_offset = false;
    try
    {
      if(is_constant_int2t(index.index))
      {
        index_offset =
          to_constant_int2t(index.index).value * type_byte_size(index.type);
        has_const_index_offset = true;
      }
    }
    catch(array_type2t::dyn_sized_array_excp *e)
    {
      // Not a constant index offset then.
    }

    object_mapt array_references;
    get_reference_set(index.source_value, array_references);

    for(const auto &a_it : array_references)
    {
      expr2tc object = object_numbering[a_it.first];

      if(is_unknown2t(object))
      {
        // Once an unknown, always an unknown.
        unknown2tc unknown(expr->type);
        insert(dest, unknown, mp_integer(0));
      }
      else
      {
        // Whatever the base object is, apply the offset represented by this
        // index expression.
        objectt o = a_it.second;

        if(has_const_index_offset && index_offset == 0)
        {
          ;
        }
        else if(has_const_index_offset && o.offset_is_zero())
        {
          o.offset = index_offset;
        }
        else
        {
          // Non constant offset -- work out what the lowest alignment is.
          // Fetch the type size of the array index element.
          mp_integer m = type_byte_size_default(index.source_value->type, 1);

          // This index operation, whatever the offset, will always multiply
          // by the size of the element type.
          unsigned int index_align = m.to_ulong();

          // Extract an offset from the old offset if set, otherwise the
          // alignment field.
          unsigned int old_align = (o.offset_is_set)
                                     ? offset2align(object, o.offset)
                                     : o.offset_alignment;

          o.offset_alignment = std::min(index_align, old_align);
          o.offset_is_set = false;
        }

        insert(dest, object, o);
      }
    }

    return;
  }
  else if(is_member2t(expr))
  {
    // The set of things referred to here are all the things the struct source
    // value may refer to, plus an additional member operation. So, fetch that
    // reference set, and add the relevant offset to the offset expr.
    const member2t &memb = to_member2t(expr);
    mp_integer offset_in_bytes;

    if(is_union_type(memb.source_value->type))
    {
      offset_in_bytes = mp_integer(0);
    }
    else
    {
      offset_in_bytes = member_offset(memb.source_value->type, memb.member);
    }

    object_mapt struct_references;
    get_reference_set(memb.source_value, struct_references);

    for(const auto &it : struct_references)
    {
      expr2tc object = object_numbering[it.first];

      // An unknown or null base is /always/ unknown or null.
      if(
        is_unknown2t(object) || is_null_object2t(object) ||
        (is_typecast2t(object) && is_null_object2t(to_typecast2t(object).from)))
      {
        unknown2tc unknown(memb.type);
        insert(dest, unknown, mp_integer(0));
      }
      else
      {
        objectt o = it.second;

        // XXX -- in terms of alignment, I believe this doesn't require
        // anything, as we're constructing an expression that takes account
        // of this. Also the same for references to indexes?
        if(o.offset_is_set)
          o.offset += offset_in_bytes;

        insert(dest, object, o);
      }
    }

    return;
  }
  else if(is_if2t(expr))
  {
    // This if expr couldn't be simplified out; take the reference set of each
    // side.
    const if2t &anif = to_if2t(expr);
    get_reference_set_rec(anif.true_value, dest);
    get_reference_set_rec(anif.false_value, dest);
    return;
  }
  else if(is_typecast2t(expr))
  {
    // Blast straight through typecasts.
    const typecast2t &cast = to_typecast2t(expr);
    get_reference_set_rec(cast.from, dest);
    return;
  }
  else if(is_byte_extract2t(expr))
  {
    // Address of byte extracts can refer to the object that is being extracted
    // from.
    const byte_extract2t &extract = to_byte_extract2t(expr);

    // This may or may not have a constant offset
    objectt o =
      (is_constant_int2t(extract.source_offset))
        ? objectt(true, to_constant_int2t(extract.source_offset).value)
        // Unclear what to do about alignments; default to nothing.
        : objectt(false, 1);

    insert(dest, extract.source_value, o);
    return;
  }
  else if(is_concat2t(expr))
  {
    const concat2t &concat = to_concat2t(expr);
    get_reference_set_rec(concat.side_1, dest);
    get_reference_set_rec(concat.side_2, dest);
    return;
  }

  // If we didn't recognize the expression, then we have no idea what this
  // refers to, so store an unknown expr.
  unknown2tc unknown(expr->type);
  insert(dest, unknown, mp_integer(0));
}

void value_sett::assign(
  const expr2tc &lhs,
  const expr2tc &rhs,
  bool add_to_sets)
{
  // Assignment interpretation.

  if(is_if2t(rhs))
  {
    // If the rhs could be either side of this if, perform the assigment of
    // either side. In case it refers to itself, assign to a temporary first,
    // then assign back.
    const if2t &ifref = to_if2t(rhs);

    // Build a sym specific to this type. Give l1 number to guard against
    // recursively entering this code path
    symbol2tc xchg_sym(
      lhs->type, xchg_name, symbol2t::level1, xchg_num++, 0, 0, 0);

    assign(xchg_sym, ifref.true_value, false);
    assign(xchg_sym, ifref.false_value, true);
    assign(lhs, xchg_sym, add_to_sets);

    erase(xchg_sym->get_symbol_name());
    return;
  }

  // Must have concrete type.
  assert(!is_symbol_type(lhs));
  const type2tc &lhs_type = lhs->type;

  if(is_struct_type(lhs_type) || is_union_type(lhs_type))
  {
    // Assign the values of all members of the rhs thing to the lhs. It's
    // sort-of-valid for the right hand side to be a superclass of the subclass,
    // in which case there are some fields not common between them, so we
    // iterate over the superclasses members.
    const std::vector<type2tc> &members = (is_struct_type(rhs->type))
                                            ? to_struct_type(rhs->type).members
                                            : to_union_type(rhs->type).members;
    const std::vector<irep_idt> &member_names =
      (is_struct_type(rhs->type)) ? to_struct_type(rhs->type).member_names
                                  : to_union_type(rhs->type).member_names;

    unsigned int i = 0;
    for(std::vector<type2tc>::const_iterator c_it = members.begin();
        c_it != members.end();
        c_it++, i++)
    {
      const type2tc &subtype = *c_it;
      const irep_idt &name = member_names[i];

      // ignore methods
      if(is_code_type(subtype))
        continue;

      member2tc lhs_member(subtype, lhs, name);

      expr2tc rhs_member;
      if(is_unknown2t(rhs))
      {
        rhs_member = unknown2tc(subtype);
      }
      else if(is_invalid2t(rhs))
      {
        rhs_member = invalid2tc(subtype);
      }
      else
      {
        assert(
          base_type_eq(rhs->type, lhs_type, ns) ||
          is_subclass_of(lhs_type, rhs->type, ns));
        expr2tc rhs_member = make_member(rhs, name);

        // XXX -- shouldn't this be one level of indentation up?
        assign(lhs_member, rhs_member, add_to_sets);
      }
    }
  }
  else if(is_array_type(lhs_type))
  {
    const array_type2t &arr_type = to_array_type(lhs_type);
    unknown2tc unknown(index_type2());
    index2tc lhs_index(arr_type.subtype, lhs, unknown);

    if(is_unknown2t(rhs) || is_invalid2t(rhs))
    {
      // Assign an uknown subtype value to the array's (unknown) index.
      unknown2tc unknown_field(arr_type.subtype);
      assign(lhs_index, unknown_field, add_to_sets);
    }
    else
    {
      assert(base_type_eq(rhs->type, lhs_type, ns));

      if(is_constant_array_of2t(rhs))
      {
        assign(lhs_index, to_constant_array_of2t(rhs).initializer, add_to_sets);
      }
      else if(is_constant_array2t(rhs) || is_constant_expr(rhs))
      {
// ...whattt
#if 0
        forall_operands(o_it, rhs)
        {
          assign(lhs_index, *o_it, add_to_sets);
          add_to_sets=true;
        }
#endif
        rhs->foreach_operand(
          [this, &add_to_sets, &lhs_index](const expr2tc &e) {
            assign(lhs_index, e, add_to_sets);
            add_to_sets = true;
          });
      }
      else if(is_with2t(rhs))
      {
        const with2t &with = to_with2t(rhs);

        unknown2tc unknown(index_type2());
        index2tc idx(arr_type.subtype, with.source_value, unknown);

        assign(lhs_index, idx, add_to_sets);
        assign(lhs_index, with.update_value, true);
      }
      else
      {
        unknown2tc unknown(index_type2());
        index2tc rhs_idx(arr_type.subtype, rhs, unknown);
        assign(lhs_index, rhs_idx, true);
      }
    }
  }
  else
  {
    // basic type
    object_mapt values_rhs;

    get_value_set(rhs, values_rhs);

    assign_rec(lhs, values_rhs, "", add_to_sets);
  }
}

void value_sett::do_free(const expr2tc &op)
{
  // op must be a pointer
  assert(is_pointer_type(op));

  // find out what it points to
  object_mapt value_set;
  get_value_set(op, value_set);

  // find out which *instances* interest us
  expr_sett to_mark;

  for(const auto &it : value_set)
  {
    const expr2tc &object = object_numbering[it.first];

    if(is_dynamic_object2t(object))
    {
      const dynamic_object2t &dynamic_object = to_dynamic_object2t(object);

      if(!dynamic_object.invalid)
      {
        to_mark.insert(dynamic_object.instance);
      }
    }
  }

  // mark these as 'may be invalid'
  // this, unfortunately, destroys the sharing
  for(auto &value : values)
  {
    object_mapt new_object_map;

    bool changed = false;

    for(object_mapt::const_iterator o_it = value.second.object_map.begin();
        o_it != value.second.object_map.end();
        o_it++)
    {
      const expr2tc &object = object_numbering[o_it->first];

      if(is_dynamic_object2t(object))
      {
        const expr2tc &instance = to_dynamic_object2t(object).instance;

        if(to_mark.count(instance) == 0)
          set(new_object_map, o_it);
        else
        {
          // adjust
          objectt o = o_it->second;
          dynamic_object2tc new_dyn(object);
          new_dyn->invalid = false;
          new_dyn->unknown = true;
          insert(new_object_map, new_dyn, o);
          changed = true;
        }
      }
      else
        set(new_object_map, o_it);
    }

    if(changed)
      value.second.object_map = new_object_map;
  }
}

void value_sett::assign_rec(
  const expr2tc &lhs,
  const object_mapt &values_rhs,
  const std::string &suffix,
  bool add_to_sets)
{
  if(is_symbol2t(lhs))
  {
    std::string identifier = to_symbol2t(lhs).get_symbol_name();

    if(add_to_sets)
      make_union(get_entry(identifier, suffix).object_map, values_rhs);
    else
      get_entry(identifier, suffix).object_map = values_rhs;
  }
  else if(is_dynamic_object2t(lhs))
  {
    const dynamic_object2t &dynamic_object = to_dynamic_object2t(lhs);

    if(is_unknown2t(dynamic_object.instance))
      return; // We're assigning to something unknown. Not much we can do.
    assert(is_constant_int2t(dynamic_object.instance));
    unsigned int idnum =
      to_constant_int2t(dynamic_object.instance).value.to_long();
    const std::string name = "value_set::dynamic_object" + i2string(idnum);

    make_union(get_entry(name, suffix).object_map, values_rhs);
  }
  else if(is_dereference2t(lhs))
  {
    object_mapt reference_set;
    get_reference_set(lhs, reference_set);

    if(reference_set.size() != 1)
      add_to_sets = true;

    for(const auto &it : reference_set)
    {
      const expr2tc obj = object_numbering[it.first];

      if(!is_unknown2t(obj))
        assign_rec(obj, values_rhs, suffix, add_to_sets);
    }
  }
  else if(is_index2t(lhs))
  {
    assert(
      is_array_type(to_index2t(lhs).source_value) ||
      is_string_type(to_index2t(lhs).source_value) ||
      is_dynamic_object2t(to_index2t(lhs).source_value));

    assign_rec(to_index2t(lhs).source_value, values_rhs, "[]" + suffix, true);
  }
  else if(is_member2t(lhs))
  {
    type2tc tmp;
    const member2t &member = to_member2t(lhs);
    const std::string &component_name = member.member.as_string();

    // Might travel through a dereference, in which case type resolving is
    // required
    const type2tc *ourtype = &member.source_value->type;
    if(is_symbol_type(*ourtype))
    {
      tmp = ns.follow(*ourtype);
      ourtype = &tmp;
    }

    assert(
      is_struct_type(*ourtype) || is_union_type(*ourtype) ||
      is_dynamic_object2t(member.source_value));

    assign_rec(
      to_member2t(lhs).source_value,
      values_rhs,
      "." + component_name + suffix,
      add_to_sets);
  }
  else if(
    is_constant_string2t(lhs) || is_null_object2t(lhs) ||
    is_valid_object2t(lhs) || is_deallocated_obj2t(lhs) ||
    is_dynamic_size2t(lhs) || is_constant_array2t(lhs))
  {
    // Ignored
  }
  else if(is_typecast2t(lhs))
  {
    assign_rec(to_typecast2t(lhs).from, values_rhs, suffix, add_to_sets);
  }
  else if(is_byte_extract2t(lhs))
  {
    assign_rec(to_byte_extract2t(lhs).source_value, values_rhs, suffix, true);
  }
  else
    throw "assign NYI: `" + get_expr_id(lhs) + "'";
}

void value_sett::do_function_call(
  const symbolt &symbol,
  const std::vector<expr2tc> &arguments)
{
  const code_typet &type = to_code_type(symbol.type);

  type2tc tmp_migrated_type;
  migrate_type(type, tmp_migrated_type);
  const code_type2t &migrated_type =
    dynamic_cast<const code_type2t &>(*tmp_migrated_type.get());

  const std::vector<type2tc> &argument_types = migrated_type.arguments;
  const std::vector<irep_idt> &argument_names = migrated_type.argument_names;

  // these first need to be assigned to dummy, temporary arguments
  // and only thereafter to the actuals, in order
  // to avoid overwriting actuals that are needed for recursive
  // calls

  for(unsigned i = 0; i < arguments.size(); i++)
  {
    const std::string identifier = "value_set::dummy_arg_" + i2string(i);
    add_var(identifier, "");

    expr2tc dummy_lhs;
    expr2tc tmp_arg = arguments[i];
    if(is_nil_expr(tmp_arg))
    {
      // As a workaround for the "--function" option, which feeds "nil"
      // arguments in here, take the expected function argument type rather
      // than the type from the argument.
      tmp_arg = unknown2tc(argument_types[i]);
      dummy_lhs = symbol2tc(argument_types[i], identifier);
    }
    else
    {
      dummy_lhs = symbol2tc(arguments[i]->type, identifier);
    }

    assign(dummy_lhs, tmp_arg, true);
  }

  // now assign to 'actual actuals'

  unsigned i = 0;

  std::vector<type2tc>::const_iterator it2 = argument_types.begin();
  for(std::vector<irep_idt>::const_iterator it = argument_names.begin();
      it != argument_names.end();
      it++, it2++)
  {
    const std::string &identifier = it->as_string();
    if(identifier == "")
      continue;

    add_var(identifier, "");

    symbol2tc v_expr(*it2, "value_set::dummy_arg_" + i2string(i));

    symbol2tc actual_lhs(*it2, identifier);
    assign(actual_lhs, v_expr, true);
    i++;
  }

  // And now delete the value set dummy args. They're going to end up
  // accumulating values from each function call that is made, which is a
  // bad plan.
  for(unsigned i = 0; i < arguments.size(); i++)
  {
    del_var("value_set::dummy_arg_" + i2string(i), "");
  }
}

void value_sett::do_end_function(const expr2tc &lhs)
{
  if(is_nil_expr(lhs))
    return;

  symbol2tc rhs(lhs->type, irep_idt("value_set::return_value"));

  assign(lhs, rhs);
}

void value_sett::apply_code(const expr2tc &code)
{
  if(is_code_block2t(code))
  {
    const code_block2t &ref = to_code_block2t(code);
    for(auto const &it : ref.operands)
      apply_code(it);
  }
  else if(is_code_assign2t(code))
  {
    const code_assign2t &ref = to_code_assign2t(code);
    assign(ref.target, ref.source);
  }
  else if(is_code_init2t(code))
  {
    const code_init2t &ref = to_code_init2t(code);
    assign(ref.target, ref.source);
  }
  else if(is_code_decl2t(code))
  {
    const code_decl2t &ref = to_code_decl2t(code);
    symbol2tc sym(ref.type, ref.value);
    invalid2tc invalid(ref.type);
    assign(sym, invalid);
  }
  else if(is_code_expression2t(code))
  {
    // can be ignored, we don't expect sideeffects here
  }
  else if(is_code_free2t(code))
  {
    // this may kill a valid bit
    const code_free2t &ref = to_code_free2t(code);
    do_free(ref.operand);
  }
  else if(is_code_printf2t(code))
  {
    // doesn't do anything
  }
  else if(is_code_return2t(code))
  {
    // this is turned into an assignment
    const code_return2t &ref = to_code_return2t(code);
    if(!is_nil_expr(ref.operand))
    {
      symbol2tc sym(ref.operand->type, "value_set::return_value");
      assign(sym, ref.operand);
    }
  }
  else if(is_code_asm2t(code))
  {
    // Ignore assembly. No idea why it isn't preprocessed out anyway.
  }
  else if(is_code_cpp_delete2t(code) || is_code_cpp_del_array2t(code))
  {
    // Ignore these too
  }
  else
  {
    std::cerr << code->pretty() << std::endl;
    std::cerr << "value_sett: unexpected statement" << std::endl;
    abort();
  }
}

expr2tc
value_sett::make_member(const expr2tc &src, const irep_idt &component_name)
{
  const type2tc &type = src->type;
  assert(is_struct_type(type) || is_union_type(type));

  // Work around for the current lack of type inheretance
  const std::vector<type2tc> &members = (is_struct_type(type))
                                          ? to_struct_type(type).members
                                          : to_union_type(type).members;

  if(is_constant_struct2t(src))
  {
    unsigned no = to_struct_type(type).get_component_number(component_name);
    return to_constant_struct2t(src).datatype_members[no];
  }
  if(is_with2t(src))
  {
    const with2t &with = to_with2t(src);
    assert(is_constant_string2t(with.update_field));
    const constant_string2t &memb_name =
      to_constant_string2t(with.update_field);

    if(component_name == memb_name.value)
      // yes! just take op2
      return with.update_value;

    // no! do this recursively
    return make_member(with.source_value, component_name);
  }
  else if(is_typecast2t(src))
  {
    // push through typecast
    return make_member(to_typecast2t(src).from, component_name);
  }

  // give up
  unsigned no = static_cast<const struct_union_data &>(*type.get())
                  .get_component_number(component_name);
  const type2tc &subtype = members[no];
  member2tc memb(subtype, src, component_name);
  return memb;
}

void value_sett::dump() const
{
  output(std::cout);
}

void value_sett::obj_numbering_ref(unsigned int num)
{
  obj_numbering_refset[num]++;
}

void value_sett::obj_numbering_deref(unsigned int num)
{
  unsigned int refcount = --obj_numbering_refset[num];
  if(refcount == 0)
  {
    object_numbering.erase(num);
    obj_numbering_refset.erase(num);
  }
}

#ifdef WITH_PYTHON
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/operators.hpp>

// Wrap call to get_value_set to just return a python list: otherwise we wind
// up having the caller spuriously allocate a value_setst::valuest, which is
// a list.
boost::python::object get_value_set_wrapper(value_sett &vs, const expr2tc &expr)
{
  using namespace boost::python;
  value_setst::valuest v;
  vs.get_value_set(expr, v);
  // Convert resulting list to a python list

  list l;
  for(const expr2tc &e : v)
    l.append(e);

  return l;
}

void build_value_set_classes()
{
  using namespace boost::python;
  {
    bool (value_sett::*insert)(
      value_sett::object_mapt &, unsigned, const value_sett::objectt &) const =
      &value_sett::insert;
    bool (value_sett::*insert_expr)(
      value_sett::object_mapt &, const expr2tc &, const mp_integer &) const =
      &value_sett::insert;
    value_sett::entryt &(value_sett::*get_entry)(const value_sett::entryt &) =
      &value_sett::get_entry;
    value_sett::entryt &(value_sett::*get_entry_named)(
      const std::string &, const std::string &) = &value_sett::get_entry;
    bool (value_sett::*make_union_objs)(
      value_sett::object_mapt &, const value_sett::object_mapt &) const =
      &value_sett::make_union;
    bool (value_sett::*make_union_values)(const value_sett::valuest &, bool) =
      &value_sett::make_union;
    bool (value_sett::*make_union_value_set)(const value_sett &, bool keepnew) =
      &value_sett::make_union;
    void (value_sett::*get_reference_set)(
      const expr2tc &, value_setst::valuest &) const =
      &value_sett::get_reference_set;

    scope foo =
      class_<value_sett>("value_set", init<namespacet>())
        .def("get_natural_alignment", &value_sett::get_natural_alignment)
        .def("offset2align", &value_sett::offset2align)
        .def("to_expr", &value_sett::to_expr)
        .def("set", &value_sett::set)
        .def("insert", insert)
        .def("insert_expr", insert_expr)
        .def("erase", &value_sett::erase)
        .def("get_value_set", get_value_set_wrapper)
        .def("clear", &value_sett::clear)
        .def("del_var", &value_sett::del_var)
        .def("get_entry", get_entry, return_internal_reference<>())
        .def("get_entry_named", get_entry_named, return_internal_reference<>())
        .def("dump", &value_sett::dump)
        .def("make_union_objs", make_union_objs)
        .def("make_union_values", make_union_values)
        .def("make_union_value_set", make_union_value_set)
        .def("apply_code", &value_sett::apply_code)
        .def("assign", &value_sett::assign)
        .def("do_function_call", &value_sett::do_function_call)
        .def("do_end_function", &value_sett::do_end_function)
        .def("get_reference_set", get_reference_set)
        .def_readwrite("object_numbering", &value_sett::object_numbering)
        .def_readwrite("values", &value_sett::values);
    // XXX object numberingt?

    class_<value_sett::valuest>("valuest").def(
      map_indexing_suite<value_sett::valuest>());

    class_<value_sett::objectt>("objectt", init<bool, unsigned int>())
      .def(init<bool, BigInt>())
      .def("offset_is_zero", &value_sett::objectt::offset_is_zero)
      .def_readwrite("offset", &value_sett::objectt::offset)
      .def_readwrite("offset_is_set", &value_sett::objectt::offset_is_set)
      .def_readwrite(
        "offset_alignment", &value_sett::objectt::offset_alignment);

    class_<value_sett::object_mapt>("object_mapt")
      .def(map_indexing_suite<value_sett::object_mapt>());

    class_<value_sett::entryt>("entryt")
      .def(init<std::string, std::string>())
      .def_readwrite("identifier", &value_sett::entryt::identifier)
      .def_readwrite("suffix", &value_sett::entryt::suffix)
      .def_readwrite("object_map", &value_sett::entryt::object_map);

    class_<object_numberingt>("object_numberingt")
      .def("number", &object_numberingt::number)
      .def("get_number", &object_numberingt::get_number);
  }
}
#endif
