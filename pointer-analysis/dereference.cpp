/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <sstream>
#include <prefix.h>
#include <expr_util.h>
#include <c_misc.h>
#include <base_type.h>
#include <arith_tools.h>
#include <rename.h>
#include <i2string.h>
#include <array_name.h>
#include <config.h>
#include <std_expr.h>
#include <cprover_prefix.h>
#include <type_byte_size.h>

#include <ansi-c/c_types.h>
#include <ansi-c/c_typecast.h>
#include <pointer-analysis/value_set.h>
#include <langapi/language_util.h>

#include "dereference.h"

// global data, horrible
unsigned int dereferencet::invalid_counter=0;

bool dereferencet::has_dereference(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  forall_operands2(it, idx, expr)
    if(has_dereference(*it))
      return true;

  if (is_dereference2t(expr) ||
     (is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value)))
    return true;

  return false;
}

const expr2tc& dereferencet::get_symbol(const expr2tc &expr)
{
  if (is_member2t(expr))
    return get_symbol(to_member2t(expr).source_value);
  else if (is_index2t(expr))
    return get_symbol(to_index2t(expr).source_value);

  return expr;
}

expr2tc
dereferencet::dereference(
  const expr2tc &src,
  const guardt &guard,
  const modet mode,
  expr2tc top_scalar_expr)
{
  expr2tc dest = src;
  assert(is_pointer_type(dest));

  // Pointers type won't have been resolved; do that now.
  pointer_type2t &dest_type = to_pointer_type(dest.get()->type);
  typet tmp_ptr_subtype = migrate_type_back(dest_type.subtype);
  const typet dereftype = ns.follow(tmp_ptr_subtype);

  migrate_type(dereftype, dest_type.subtype);

  // save the dest for later, dest might be destroyed
  const expr2tc deref_expr(dest);

  // type of the object
  const type2tc &type = dest_type.subtype;

  // collect objects dest may point to
  value_setst::valuest points_to_set;

  dereference_callback.get_value_set(dest, points_to_set);

  // now build big case split
  // only "good" objects

  expr2tc value;

  for(value_setst::valuest::const_iterator
      it=points_to_set.begin();
      it!=points_to_set.end();
      it++)
  {
    expr2tc new_value, pointer_guard;

    build_reference_to(*it, mode, dest, type, new_value, pointer_guard, guard,
                       top_scalar_expr);

    if (!is_nil_expr(new_value))
    {
      if (is_nil_expr(value)) {
        value = new_value;
      } else {
        // Chain a big if-then-else case.
        value = if2tc(type, pointer_guard, new_value, value);
      }
    }
  }

  if (is_nil_expr(value))
  {
    // first see if we have a "failed object" for this pointer

    const symbolt *failed_symbol;

    if (dereference_callback.has_failed_symbol(deref_expr, failed_symbol))
    {
      // yes!
      exprt tmp_val = symbol_expr(*failed_symbol);
      migrate_expr(tmp_val, value);
    }
    else
    {
      // else, do new symbol

      symbolt symbol;
      symbol.name="symex::invalid_object"+i2string(invalid_counter++);
      symbol.base_name="invalid_object";
      symbol.type=migrate_type_back(type);

      // make it a lvalue, so we can assign to it
      symbol.lvalue=true;

      get_new_name(symbol, ns);

      exprt tmp_sym_expr = symbol_expr(symbol);

      new_context.move(symbol);

      // Due to migration hiccups, migration must occur after the symbol
      // appears in the symbol table.
      migrate_expr(tmp_sym_expr, value);
    }
  }

  dest = value;
  return dest;
}

bool dereferencet::dereference_type_compare(
  expr2tc &object, const type2tc &dereference_type) const
{
  const type2tc object_type = object->type;

  if (is_empty_type(dereference_type))
    return true; // always ok

  if (base_type_eq(object_type, dereference_type, ns)) {
    // Ok, they just match. However, the SMT solver that receives this formula
    // in the end may object to taking an equivalent type and instead demand
    // that the types are exactly the same. So, slip in a typecast.
    object = typecast2tc(dereference_type, object);
    return true;
  }

  // Check for C++ subclasses; we can cast derived up to base safely.
  if (is_struct_type(object) && is_struct_type(dereference_type)) {
    if (is_subclass_of(object->type, dereference_type, ns)) {
      object = typecast2tc(dereference_type, object);
      return true;
    }
  }

  // check for struct prefixes

  type2tc ot_base(object_type), dt_base(dereference_type);

  base_type(ot_base, ns);
  base_type(dt_base, ns);

  if (is_struct_type(ot_base) && is_struct_type(dt_base))
  {
    typet tmp_ot_base = migrate_type_back(ot_base);
    typet tmp_dt_base = migrate_type_back(dt_base);
    if (to_struct_type(tmp_dt_base).is_prefix_of(
         to_struct_type(tmp_ot_base)))
    {
      object = typecast2tc(dereference_type, object);
      return true; // ok, dt is a prefix of ot
    }
  }

  // we are generous about code pointers
  if (is_code_type(dereference_type) && is_code_type(object_type))
    return true;

  // really different

  return false;
}

void dereferencet::build_reference_to(
  const expr2tc &what,
  const modet mode,
  const expr2tc &deref_expr,
  const type2tc &type,
  expr2tc &value,
  expr2tc &pointer_guard,
  const guardt &guard,
  const expr2tc &top_scalar_expr __attribute__((unused)))
{
  value = expr2tc();
  pointer_guard = false_expr;

  if (is_unknown2t(what) || is_invalid2t(what))
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is an invalid pointer

      invalid_pointer2tc invalid_pointer_expr(deref_expr);

      // produce new guard

      guardt tmp_guard(guard);
      tmp_guard.move(invalid_pointer_expr);
      dereference_callback.dereference_failure(
        "pointer dereference",
        "invalid pointer",
        tmp_guard);
    }

    return;
  }

  if (!is_object_descriptor2t(what)) {
    std::cerr << "unknown points-to: " << get_expr_id(what);
    abort();
  }

  const object_descriptor2t &o = to_object_descriptor2t(what);

  const expr2tc &root_object = o.get_root_object();
  const expr2tc &object = o.object;

  if (is_null_object2t(root_object))
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      type2tc nullptrtype = type2tc(new pointer_type2t(type));
      symbol2tc null_ptr(nullptrtype, "NULL");

      same_object2tc pointer_guard(deref_expr, null_ptr);

      guardt tmp_guard(guard);
      tmp_guard.add(pointer_guard);

      dereference_callback.dereference_failure(
        "pointer dereference",
        "NULL pointer", tmp_guard);
    }

    // Don't build a reference to this. You can't actually access NULL, and the
    // solver will only get confused.
    return;
  }
  else if (is_dynamic_object2t(root_object))
  {
    const dynamic_object2t &dyn_obj = to_dynamic_object2t(root_object);

    value = dereference2tc(type, deref_expr);

    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is a dynamic object

      type2tc arr_type = type2tc(new array_type2t(get_bool_type(),
                                                  expr2tc(), true));
      const symbolt *sp;
      irep_idt dyn_name = (!ns.lookup(irep_idt("c::__ESBMC_alloc"), sp))
        ? "c::__ESBMC_is_dynamic" : "cpp::__ESBMC_is_dynamic";
      symbol2tc sym(arr_type, dyn_name);
      pointer_object2tc ptr_obj(int_type2(), deref_expr);
      index2tc is_dyn_obj(get_bool_type(), sym, ptr_obj);

      if (dyn_obj.invalid || dyn_obj.unknown)
      {
        // check if it is still alive
        valid_object2tc valid_expr(deref_expr);
        not2tc not_valid_expr(valid_expr);

        guardt tmp_guard(guard);
        tmp_guard.add(is_dyn_obj);
        tmp_guard.move(not_valid_expr);
        dereference_callback.dereference_failure(
          "pointer dereference",
          "invalidated dynamic object",
          tmp_guard);
      }

      if (!options.get_bool_option("no-bounds-check") &&
              (!is_constant_int2t(o.offset) ||
               !to_constant_int2t(o.offset).constant_value.is_zero()))
      {
        {
          // check lower bound
          pointer_offset2tc obj_offset(index_type2(), deref_expr);
          lessthan2tc lt(obj_offset, zero_int);

          guardt tmp_guard(guard);
          tmp_guard.add(is_dyn_obj);
          tmp_guard.move(lt);
          dereference_callback.dereference_failure(
            "pointer dereference",
            "dynamic object lower bound", tmp_guard);
        }

        {
          // check upper bound
          //nec: ex37.c
          dynamic_size2tc size_expr(deref_expr);

          expr2tc obj_offs = pointer_offset2tc(index_type2(), deref_expr);
          obj_offs = typecast2tc(int_type2(), obj_offs);
          lessthanequal2tc lte(size_expr, obj_offs);

          guardt tmp_guard(guard);
          tmp_guard.add(is_dyn_obj);
          tmp_guard.move(lte);

          dereference_callback.dereference_failure(
            "pointer dereference",
            "dynamic object upper bound", tmp_guard);
        }
      }
    }
  }
  else
  {
    value = object;

    type2tc ptr_type = type2tc(new pointer_type2t(object->type));
    address_of2tc obj_ptr(ptr_type, object);

    pointer_guard = same_object2tc(deref_expr, obj_ptr);

    guardt tmp_guard(guard);
    tmp_guard.add(pointer_guard);

    valid_check(object, tmp_guard, mode);

    if (is_constant_expr(o.offset)) {

      // See whether or not we need to munge the object into the desired type;
      // this will return false if we need to juggle the type in a significant
      // way, true if they're either the same type or extremely similar. value
      // may be replaced with a typecast.
      expr2tc orig_value = value;
      if (!dereference_type_compare(value, type))
      {
        // Not a compatible thing; stitch it together in the memory model.
        construct_from_dyn_offset(value, o.offset, type, guard);
      }

      const constant_int2t &theint = to_constant_int2t(o.offset);
      if (theint.constant_value.to_ulong() == 0)
        construct_from_zero_offset(value, type, tmp_guard);
      else
        construct_from_const_offset(value, o.offset, type, tmp_guard);
    } else {
      expr2tc offset = pointer_offset2tc(index_type2(), deref_expr);
      construct_from_dyn_offset(value, offset, type, tmp_guard);
    }
  }
}

void
dereferencet::construct_from_zero_offset(expr2tc &value, const type2tc &type,
                                          const guardt &guard)
{

  expr2tc orig_value = get_base_object(value);

  if (is_scalar_type(orig_value)) {
    // dereference_type_compare will have slipped in a typecast.
  } else if (is_array_type(orig_value)) {
    // We have zero offset. Just select things out.
    const array_type2t &arr = to_array_type(value->type);
    assert(!is_array_type(arr.subtype) && "Can't cope with multidimensional arrays right now captain1");
    assert(!is_structure_type(arr.subtype) && "Also not considering arrays of structs at this time, sorry");

    unsigned int access_size_int = type->get_width() / 8;
    unsigned long subtype_size_int = type_byte_size(*arr.subtype).to_ulong();

    bounds_check(orig_value->type, zero_int, access_size_int, guard);

    // Now, if the subtype size is >= the read size, we can just either cast
    // or extract out. If not, we have to extract by conjoining elements.
    if (!is_big_endian && subtype_size_int >= access_size_int) {
      // Voila, one can just select and cast. This works because little endian
      // just allows for this to happen.
      index2tc idx(arr.subtype, orig_value, zero_uint);
      typecast2tc cast(type, idx);
      value = cast;
    } else {
      // Nope, one must byte extract this.
      const type2tc &bytetype = get_uint8_type();
      value = byte_extract2tc(bytetype, orig_value, zero_uint, is_big_endian);
      if (type != bytetype)
        value = typecast2tc(type, value);
    }
  } else {
    assert(0 && "Dereferencing zero offset to a struct?");
  }
}

void
dereferencet::construct_from_const_offset(expr2tc &value, const expr2tc &offset,
                                          const type2tc &type,
                                          const guardt &guard)
{

  // XXX This isn't taking account of the additional offset being torn through
  expr2tc base_object = get_base_object(value);

  const constant_int2t &theint = to_constant_int2t(offset);
  const type2tc &bytetype = get_uint8_type();
  value = byte_extract2tc(bytetype, base_object, offset, is_big_endian);

  unsigned long access_sz =  type_byte_size(*type).to_ulong();
  if (is_array_type(base_object) || is_string_type(base_object)) {
    bounds_check(base_object->type, offset, access_sz, guard);
  } else {
    unsigned long sz = type_byte_size(*value->type).to_ulong();
    if (sz + access_sz > theint.constant_value.to_ulong()) {
      if(!options.get_bool_option("no-pointer-check")) {
        guardt tmp_guard2(guard);
        tmp_guard2.add(false_expr);

        dereference_callback.dereference_failure(
          "pointer dereference",
          "Offset out of bounds", tmp_guard2);
      }
    }
  }
}

void
dereferencet::construct_from_dyn_offset(expr2tc &value, const expr2tc &offset,
                                        const type2tc &type,
                                        const guardt &guard)
{

  expr2tc new_offset = offset;
  if (memory_model(value, type, guard, new_offset))
  {
    // ok
  }
  else
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      //nec: ex29
      if (    (is_pointer_type(type) &&
               is_empty_type(to_pointer_type(type).subtype))
          ||
              (is_pointer_type(value) &&
               is_empty_type(to_pointer_type(value->type).subtype)))
        return;

      std::string msg="memory model not applicable (got `";
      msg+=from_type(ns, "", value->type);
      msg+="', expected `";
      msg+=from_type(ns, "", type);
      msg+="')";

      dereference_callback.dereference_failure(
        "pointer dereference",
        msg, guard);
    }

    value = expr2tc();
    return; // give up, no way that this is ok
  }
}

void dereferencet::valid_check(
  const expr2tc &object,
  const guardt &guard,
  const modet mode)
{
  if(options.get_bool_option("no-pointer-check"))
    return;

  if(mode==FREE)
  {
    dereference_callback.dereference_failure(
      "pointer dereference",
      "free() of non-dynamic memory",
      guard);
    return;
  }

  const expr2tc &symbol = get_symbol(object);

  if (is_constant_string2t(symbol))
  {
    // always valid, but can't write

    if(mode==WRITE)
    {
      dereference_callback.dereference_failure(
        "pointer dereference",
        "write access to string constant",
        guard);
    }
  }
  else if (is_nil_expr(symbol))
  {
    // always "valid", shut up
    return;
  }
  else if (is_symbol2t(symbol))
  {
    // Hacks, but as dereferencet object isn't persistent, necessary. Fix by
    // making dereferencet persistent.
    if (has_prefix(to_symbol2t(symbol).thename.as_string(), "symex::invalid_object"))
      return;

    if (dereference_callback.is_valid_object(to_symbol2t(symbol).thename))
      return; // always ok
  }
}

void dereferencet::bounds_check(const type2tc &type, const expr2tc &offset,
                                unsigned int access_size, const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  assert(is_array_type(type) || is_string_type(type));

  // Dance around getting the array type normalised.
  type2tc new_string_type;
  const array_type2t *arr_type_p = NULL;
  if (is_array_type(type)) {
    arr_type_p = &to_array_type(type);
  } else {
    const string_type2t &str_type = to_string_type(type);
    expr2tc str_size = gen_uint(str_type.width);
    new_string_type =
      type2tc(new array_type2t(get_uint8_type(), str_size, false));
    arr_type_p = &to_array_type(new_string_type);
  }

  const array_type2t &arr_type = *arr_type_p;

  // XXX --  arrays were assigned names, but we're skipping that for the moment
  // std::string name = array_name(ns, expr.source_value);

  // Firstly, bail if this is an infinite sized array. There are no bounds
  // checks to be performed.
  if (arr_type.size_is_infinite)
    return;

  // Secondly, try to calc the size of the array.
  unsigned long subtype_size_int = type_byte_size(*arr_type.subtype).to_ulong();
  constant_int2tc subtype_size(get_int32_type(), BigInt(subtype_size_int));
  mul2tc arrsize(get_uint32_type(), arr_type.array_size, subtype_size);

  // Then, expressions as to whether the access is over or under the array
  // size.
  constant_int2tc access_size_e(get_int32_type(), BigInt(access_size));
  add2tc upper_byte(get_int32_type(), offset, access_size_e);
  expr2tc lower_byte = offset;

  lessthanequal2tc le(upper_byte, arrsize);
  greaterthanequal2tc ge(lower_byte, zero_int);

  // Report these as assertions; they'll be simplified away if they're constant

  guardt tmp_guard1(guard);
  tmp_guard1.move(le);
  dereference_callback.dereference_failure("array bounds", "array upper bound",
                                           tmp_guard1);

  guardt tmp_guard2(guard);
  tmp_guard2.move(ge);
  dereference_callback.dereference_failure("array bounds", "array upper bound",
                                           tmp_guard1);
}

bool dereferencet::memory_model(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  // we will allow more or less arbitrary pointer type cast

  const type2tc &from_type = value->type;

  // first, check if it's really just a conversion

  if (is_bv_type(from_type) && is_bv_type(to_type) &&
      from_type->get_width() == to_type->get_width() &&
      is_constant_int2t(new_offset) &&
      to_constant_int2t(new_offset).constant_value.is_zero()) {
    value = typecast2tc(to_type, value);
    return true;
  }

  // otherwise, we will stich it together from bytes

  bool ret = memory_model_bytes(value, to_type, guard, new_offset);
  return ret;
}

bool dereferencet::memory_model_bytes(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  const expr2tc orig_value = value;
  const type2tc from_type = value->type;

  // Accessing code is incorrect; The C spec says that the code and data address
  // spaces should be considered seperate (i.e., Harvard arch) and so accessing
  // code via a pointer is never valid. Even though you /can/ do it on X86.
  if (is_code_type(from_type) || is_code_type(to_type)) {
    guardt tmp_guard(guard);
    dereference_callback.dereference_failure("Code seperation",
        "Dereference accesses code / program text", tmp_guard);
    return true;
  }

  assert(config.ansi_c.endianess != configt::ansi_ct::NO_ENDIANESS);

  // We allow reading more or less anything as bit-vector.
  if (is_bv_type(to_type) || is_pointer_type(to_type) ||
        is_fixedbv_type(to_type))
  {
    // Take existing pointer offset, add to the pointer offset produced by
    // this dereference. It'll get simplified at some point in the future.
    new_offset = add2tc(new_offset->type, new_offset,
                        compute_pointer_offset(value));
    expr2tc tmp = new_offset->simplify();
    if (!is_nil_expr(tmp))
      new_offset = tmp;

    // XXX This isn't taking account of the additional offset being torn through
    expr2tc base_object = get_base_object(value);


    const type2tc &bytetype = get_uint8_type();
    value = byte_extract2tc(bytetype, base_object, new_offset, is_big_endian);

    // XXX jmorse - temporary, while byte extract is still covered in bees.
    value = typecast2tc(to_type, value);


    if (!is_constant_int2t(new_offset) ||
        !to_constant_int2t(new_offset).constant_value.is_zero())
    {
      if(!options.get_bool_option("no-pointer-check"))
      {
        // Get total size of the data object we're working on.
        expr2tc total_size;
        try {
          total_size = constant_int2tc(uint_type2(),
                                       base_object->type->get_width() / 8);
        } catch (array_type2t::dyn_sized_array_excp *e) {
          expr2tc eight = gen_uint(8);
          total_size = div2tc(uint_type2(), e->size, eight);
        }

        unsigned long width = to_type->get_width() / 8;
        expr2tc const_val = gen_uint(width);
        add2tc upper_bound(uint_type2(), new_offset, const_val);
        greaterthan2tc upper_bound_eq(upper_bound, total_size);

        guardt tmp_guard(guard);
        tmp_guard.move(upper_bound_eq);
        dereference_callback.dereference_failure(
            "byte model object boundries",
            "byte access upper bound", tmp_guard);

        lessthan2tc offs_lower_bound(new_offset, zero_int);

        guardt tmp_guard2(guard);
        tmp_guard2.move(offs_lower_bound);
        dereference_callback.dereference_failure(
          "byte model object boundries",
          "word offset lower bound", tmp_guard);
      }
    }

    return true;
  }

  return false;
}
