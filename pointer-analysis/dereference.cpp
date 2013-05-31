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
#include <pointer_offset_size.h>

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

void dereferencet::dereference(
  expr2tc &dest,
  const guardt &guard,
  const modet mode)
{
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

  // if it's empty, we have a problem
  //lucas: nec: ex33.c
#if 0
  if(points_to_set.empty())
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      dereference_callback.dereference_failure(
        "pointer dereference",
        "invalid pointer", guard);
    }
  }
#endif
  for(value_setst::valuest::const_iterator
      it=points_to_set.begin();
      it!=points_to_set.end();
      it++)
  {
    expr2tc new_value, pointer_guard;

    build_reference_to(*it, mode, dest, type, new_value, pointer_guard, guard);

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
      migrate_expr(tmp_sym_expr, value);

      new_context.move(symbol);
    }
  }

  dest = value;
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
    if (is_subclass_of(object->type, dereference_type, ns))
      return true;
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
  const guardt &guard)
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

#if 1
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
#endif
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

    expr2tc offset;

    if (is_constant_expr(o.offset))
      offset = o.offset;
    else
    {
      pointer_offset2tc ptr_offs(index_type2(), deref_expr);
      pointer_offset2tc base(index_type2(), obj_ptr);

      // need to subtract base address
      offset = sub2tc(index_type2(), ptr_offs, base);
    }

    // See whether or not we need to munge the object into the desired type;
    // this will return false if we need to juggle the type in a significant
    // way, true if they're either the same type or extremely similar. value
    // may be replaced with a typecast.
    expr2tc orig_value = value;
    if (!dereference_type_compare(value, type))
    {
      if (memory_model(value, type, tmp_guard, offset))
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
            msg, tmp_guard);
        }

        value = expr2tc();
        return; // give up, no way that this is ok
      }
    }
    else
    {
      // The dereference types match closely enough; make some bounds checks
      // on the base object, not the possibly typecasted object.
      if (is_index2t(orig_value))
      {
        // So; we're working on an index, which might be wrapped in a typecast.
        // Update the offset; then encode a bounds check.
        if (is_typecast2t(value)) {
          typecast2t &cast = to_typecast2t(value);
          index2t &idx = to_index2t(cast.from);
          idx.index = offset;
          bounds_check(idx, tmp_guard);
        } else {
          index2t &idx = to_index2t(value);
          idx.index = offset;
          bounds_check(idx, tmp_guard);
        }
      }
      else if (!is_constant_int2t(offset) ||
               !to_constant_int2t(offset).constant_value.is_zero())
      {
        if(!options.get_bool_option("no-pointer-check"))
        {
          notequal2tc offs_is_not_zero(offset, zero_int);

          guardt tmp_guard2(guard);
          tmp_guard2.move(offs_is_not_zero);

          dereference_callback.dereference_failure(
            "pointer dereference",
            "offset not zero (non-array-object)", tmp_guard2);
        }
      }
    }
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

void dereferencet::bounds_check(
  const index2t &expr,
  const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  assert(is_array_type(expr.source_value) ||
         is_string_type(expr.source_value));

  std::string name = array_name(ns, expr.source_value);

  {
    if (is_constant_int2t(expr.index) &&
        !to_constant_int2t(expr.index).constant_value.is_negative())
    {
      ;
    }
    else
    {
      lessthan2tc lt(expr.index, zero_int);

      guardt tmp_guard(guard);
      tmp_guard.move(lt);
      dereference_callback.dereference_failure(
        "array bounds",
        "`"+name+"' lower bound", tmp_guard);
    }
  }

  expr2tc arr_size;
  if (is_array_type(expr.source_value)) {
    if (to_array_type(expr.source_value->type).size_is_infinite)
      // Can't overflow an infinitely sized array
      return;

    arr_size = to_array_type(expr.source_value->type).array_size;
  } else {
    expr2tc tmp_str_arr = to_constant_string2t(expr.source_value).to_array();
    arr_size = to_array_type(tmp_str_arr->type).array_size;
  }

  if (is_index2t(expr.source_value))
  {
    const index2t &index = to_index2t(expr.source_value);
    const array_type2t &arr_type_2 = to_array_type(index.source_value->type);

    assert(!arr_type_2.size_is_infinite);
    arr_size = mul2tc(index_type2(), arr_size, arr_type_2.array_size);
  }

  // Irritating - I don't know what c_implicit_typecast does, and it modifies
  // tmp_op0 it appears.
  exprt tmp_op0 = migrate_expr_back(expr.index);
  exprt tmp_op1 = migrate_expr_back(arr_size);
  if (c_implicit_typecast(tmp_op0, tmp_op1.type(), ns)) {
    std::cerr << "index address of wrong type in bounds_check" << std::endl;
    abort();
  }

  expr2tc new_index;
  migrate_expr(tmp_op0, new_index);
  greaterthanequal2tc gte(new_index, arr_size);

  guardt tmp_guard(guard);
  tmp_guard.move(gte);

  dereference_callback.dereference_failure(
    "array bounds",
    "`"+name+"' upper bound", tmp_guard);
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

  if (is_number_type(from_type) && is_number_type(to_type) &&
      from_type->get_width() == to_type->get_width())
    return memory_model_conversion(value, to_type, guard, new_offset);

  // otherwise, we will stich it together from bytes

  bool ret = memory_model_bytes(value, to_type, guard, new_offset);
  return ret;
}

bool dereferencet::memory_model_conversion(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  const type2tc from_type = value->type;

  // avoid semantic conversion in case of
  // cast to float
  if (is_fixedbv_type(to_type))
  {
    value = expr2tc(new typecast2t(to_type, value));
  }
  else
  {
    // only doing type conversion
    // just do the typecast
    value = typecast2tc(to_type, value);
  }

  // also assert that offset is zero

  if(!options.get_bool_option("no-pointer-check"))
  {
    expr2tc zero = constant_int2tc(new_offset->type, BigInt(0));
    expr2tc offs_not_zero = notequal2tc(new_offset, zero);

    guardt tmp_guard(guard);
    tmp_guard.move(offs_not_zero);
    dereference_callback.dereference_failure(
      "word bounds",
      "offset not zero", tmp_guard);
  }

  return true;
}

bool dereferencet::memory_model_bytes(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  const expr2tc orig_value = value;
  const type2tc from_type = value->type;

  // we won't try to convert to/from code
  if (is_code_type(from_type) || is_code_type(to_type))
    return false;

  // won't do this without a committment to an endianess
  if(config.ansi_c.endianess==configt::ansi_ct::NO_ENDIANESS)
    return false;

  // But anything else we will try!

  // We allow reading more or less anything as bit-vector.
  if (is_bv_type(to_type))
  {
    bool is_big_endian =
      (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);

    // Byte extract currently produced one byte, regardless of given type.
    value = byte_extract2tc(type_pool.get_uint(8), value, new_offset,
                            is_big_endian);

    // XXX jmorse - upcast the extracted byte to whatever type we're supposed to
    // have. In a correct world, we'd be stitching together the type from a
    // series of extracted bytes.
    if (to_type->get_width() != 8)
      value = typecast2tc(to_type, value);

    if (!is_constant_int2t(new_offset) ||
        !to_constant_int2t(new_offset).constant_value.is_zero())
    {
      if(!options.get_bool_option("no-pointer-check"))
      {
        unsigned long width = orig_value->type->get_width();
        expr2tc const_val = constant_int2tc(new_offset->type, BigInt(width));
        expr2tc offs_upper_bound = greaterthanequal2tc(new_offset, const_val);

        guardt tmp_guard(guard);
        tmp_guard.move(offs_upper_bound);
        dereference_callback.dereference_failure(
          "word bounds",
          "word offset upper bound", tmp_guard);
      }

      if(!options.get_bool_option("no-pointer-check"))
      {
        expr2tc zero = constant_int2tc(new_offset->type, BigInt(0));
        expr2tc offs_lower_bound = lessthan2tc(new_offset, zero);

        guardt tmp_guard(guard);
        tmp_guard.move(offs_lower_bound);
        dereference_callback.dereference_failure(
          "word bounds",
          "word offset lower bound", tmp_guard);
      }
    }

    return true;
  }

  return false;
}
