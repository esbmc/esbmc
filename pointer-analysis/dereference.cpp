/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <sstream>
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

/*******************************************************************\

Function: dereferencet::has_dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool dereferencet::has_dereference(const exprt &expr) const
{
  forall_operands(it, expr)
    if(has_dereference(*it))
      return true;

  if(expr.id()=="dereference" ||
     expr.id()=="implicit_dereference" ||
     (expr.id()=="index" && expr.operands().size()==2 &&
      expr.op0().type().id()=="pointer"))
    return true;

  return false;
}

/*******************************************************************\

Function: dereferencet::get_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const exprt& dereferencet::get_symbol(const exprt &expr)
{
  if(expr.id()=="member" || expr.id()=="index")
    return get_symbol(expr.op0());

  return expr;
}

/*******************************************************************\

Function: dereferencet::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereferencet::dereference(
  exprt &dest,
  const guardt &guard,
  const modet mode)
{
  if(dest.type().id()!="pointer")
    throw "dereference expected pointer type, but got "+
          dest.type().pretty();

  // Pointers type won't have been resolved; do that now.
  const typet dereftype = ns.follow(dest.type().subtype());
  dest.type().subtype() = dereftype;

  // save the dest for later, dest might be destroyed
  const exprt deref_expr(dest);

  // type of the object
  const typet &type=deref_expr.type().subtype();

  // collect objects dest may point to
  value_setst::valuest points_to_set;

  dereference_callback.get_value_set(dest, points_to_set);

  // now build big case split
  // only "good" objects

  exprt value;
  value.make_nil();

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
    exprt new_value, pointer_guard;

    expr2tc tmp_new_val, tmp_ptr_guard, tmp_dest;
    type2tc tmp_type;
    migrate_type(type, tmp_type);
    migrate_expr(dest, tmp_dest);
    build_reference_to(
      *it, mode, tmp_dest, tmp_type,
      tmp_new_val, tmp_ptr_guard, guard);
    new_value = migrate_expr_back(tmp_new_val);
    pointer_guard = migrate_expr_back(tmp_ptr_guard);

    if(new_value.is_not_nil())
    {
      if(value.is_nil())
        value.swap(new_value);
      else
      {
        if_exprt tmp;
        tmp.type()=type;
        tmp.cond()=pointer_guard;
        tmp.true_case()=new_value;
        tmp.false_case().swap(value);
        value.swap(tmp);
      }
    }
  }

  if(value.is_nil())
  {
    // first see if we have a "failed object" for this pointer

    const symbolt *failed_symbol;

    if(dereference_callback.has_failed_symbol(deref_expr, failed_symbol))
    {
      // yes!
      value=symbol_expr(*failed_symbol);
    }
    else
    {
      // else, do new symbol

      symbolt symbol;
      symbol.name="symex::invalid_object"+i2string(invalid_counter++);
      symbol.base_name="invalid_object";
      symbol.type=type;

      // make it a lvalue, so we can assign to it
      symbol.lvalue=true;

      get_new_name(symbol, ns);

      value=symbol_expr(symbol);

      new_context.move(symbol);
    }

    value.invalid_object(true);
  }

  dest.swap(value);
}

/*******************************************************************\

Function: dereferencet::add_checks

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereferencet::add_checks(
  const exprt &dest,
  const guardt &guard,
  const modet mode)
{
  if(dest.type().id()!="pointer")
    throw "dereference expected pointer type, but got "+
          dest.type().pretty();

  const typet &type=dest.type().subtype();

  // collect objects dest may point to
  value_setst::valuest points_to_set;

  dereference_callback.get_value_set(dest, points_to_set);

  // if it's empty, we have a problem
  if(points_to_set.empty())
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      dereference_callback.dereference_failure(
        "pointer dereference",
        "invalid pointer", guard);
    }
  }
  else
  {
    for(value_setst::valuest::const_iterator
        it=points_to_set.begin();
        it!=points_to_set.end();
        it++)
    {
      expr2tc new_value, pointer_guard, tmp_dest;
      type2tc tmp_type;
      migrate_type(type, tmp_type);
      migrate_expr(dest, tmp_dest);
      build_reference_to(
        *it, mode, tmp_dest, tmp_type,
        new_value, pointer_guard, guard);
    }
  }
}

/*******************************************************************\

Function: dereferencet::dereference_type_compare

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool dereferencet::dereference_type_compare(
  exprt &object,
  const typet &dereference_type) const
{
  const typet &object_type=object.type();

  if(dereference_type.id()=="empty")
    return true; // always ok

  if(base_type_eq(object_type, dereference_type, ns))
    return true; // ok, they just match

  // check for struct prefixes

  typet ot_base(object_type),
        dt_base(dereference_type);

  base_type(ot_base, ns);
  base_type(dt_base, ns);

  if(ot_base.id()=="struct" &&
     dt_base.id()=="struct")
  {
    if(to_struct_type(dt_base).is_prefix_of(
         to_struct_type(ot_base)))
    {
      object.make_typecast(dereference_type);
      return true; // ok, dt is a prefix of ot
    }
  }

  // we are generous about code pointers
  if(dereference_type.is_code() &&
     object_type.is_code())
    return true;

  // really different

  return false;
}

/*******************************************************************\

Function: dereferencet::build_reference_to

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
  pointer_guard = expr2tc(new constant_bool2t(false));

  if (is_unknown2t(what) || is_invalid2t(what))
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is an invalid pointer

      expr2tc invalid_pointer_expr = expr2tc(new invalid_pointer2t(deref_expr));

      // produce new guard

      guardt tmp_guard(guard);
      exprt tmp_guard_expr = migrate_expr_back(invalid_pointer_expr);
      tmp_guard.move(tmp_guard_expr);
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
      expr2tc null_ptr = expr2tc(new symbol2t(nullptrtype, "NULL"));

      expr2tc pointer_guard = expr2tc(new same_object2t(deref_expr, null_ptr));

      guardt tmp_guard(guard);
      exprt tmpback = migrate_expr_back(pointer_guard);
      tmp_guard.add(tmpback);

      dereference_callback.dereference_failure(
        "pointer dereference",
        "NULL pointer", tmp_guard);
    }
  }
  else if (is_dynamic_object2t(root_object))
  {
    const dynamic_object2t &dyn_obj = to_dynamic_object2t(root_object);

    value = expr2tc(new dereference2t(type, deref_expr));

    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is a dynamic object

      type2tc arr_type = type2tc(new array_type2t(type_pool.get_bool(),
                                                  expr2tc(), true));
      expr2tc sym = expr2tc(new symbol2t(arr_type, "c::__ESBMC_is_dynamic"));
      expr2tc ptr_obj = expr2tc(new pointer_object2t(int_type2(), deref_expr));
      expr2tc is_dyn_obj = expr2tc(new index2t(type_pool.get_bool(), sym,
                                               ptr_obj));

      if (dyn_obj.invalid || dyn_obj.unknown)
      {
        // check if it is still alive
        expr2tc valid_expr = expr2tc(new valid_object2t(deref_expr));
        expr2tc not_valid_expr = expr2tc(new not2t(valid_expr));

        guardt tmp_guard(guard);
        exprt tmp_is_dyn = migrate_expr_back(is_dyn_obj);
        exprt tmp_not_valid = migrate_expr_back(not_valid_expr);
        tmp_guard.add(tmp_is_dyn);
        tmp_guard.move(tmp_not_valid);
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
          expr2tc zero = expr2tc(new constant_int2t(index_type2(), 0));
          expr2tc obj_offset = expr2tc(new pointer_offset2t(index_type2(),
                                                            deref_expr));

          expr2tc lt = expr2tc(new lessthan2t(obj_offset, zero));

          guardt tmp_guard(guard);
          exprt tmp_is_dyn_obj = migrate_expr_back(is_dyn_obj);
          exprt tmp_lt = migrate_expr_back(lt);
          tmp_guard.add(tmp_is_dyn_obj);
          tmp_guard.move(tmp_lt);
          dereference_callback.dereference_failure(
            "pointer dereference",
            "dynamic object lower bound", tmp_guard);
        }

        {
          // check upper bound
          //nec: ex37.c
          expr2tc size_expr = expr2tc(new dynamic_size2t(deref_expr));

          expr2tc obj_offs = expr2tc(new pointer_offset2t(index_type2(),
                                                          deref_expr));
          obj_offs = expr2tc(new typecast2t(int_type2(), obj_offs));

          expr2tc lte = expr2tc(new lessthanequal2t(size_expr, obj_offs));

          guardt tmp_guard(guard);
          exprt tmp_is_dyn_obj = migrate_expr_back(is_dyn_obj);
          exprt tmp_lte = migrate_expr_back(lte);
          tmp_guard.add(tmp_is_dyn_obj);
          tmp_guard.move(tmp_lte);

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
    expr2tc obj_ptr = expr2tc(new address_of2t(ptr_type, object));

    pointer_guard = expr2tc(new same_object2t(deref_expr, obj_ptr));

    guardt tmp_guard(guard);
    exprt tmp_ptr_guard = migrate_expr_back(pointer_guard);
    tmp_guard.add(tmp_ptr_guard);

    exprt tmp_obj = migrate_expr_back(object);
    valid_check(tmp_obj, tmp_guard, mode);

    expr2tc offset;

    if (is_constant_expr(o.offset))
      offset = o.offset;
    else
    {
      expr2tc ptr_offs = expr2tc(new pointer_offset2t(index_type2(),
                                                      deref_expr));
      expr2tc base = expr2tc(new pointer_offset2t(index_type2(), obj_ptr));

      // need to subtract base address
      offset = expr2tc(new sub2t(index_type2(), ptr_offs, base));
    }

    exprt tmp_value = migrate_expr_back(value);
    typet tmp_type = migrate_type_back(type);
    if (!dereference_type_compare(tmp_value, tmp_type))
    {
      exprt tmp_offset = migrate_expr_back(offset);
      if (memory_model(tmp_value, tmp_type, tmp_guard, tmp_offset))
      {
        migrate_expr(tmp_value, value);
        migrate_expr(tmp_offset, offset);
        // ok
      }
      else
      {
        if(!options.get_bool_option("no-pointer-check"))
        {
          //nec: ex29
          const pointer_type2t &val_ptr_type = to_pointer_type(value->type);
          const pointer_type2t &ptr_type = to_pointer_type(type);
          if (is_empty_type(ptr_type.subtype) ||
              is_empty_type(val_ptr_type.subtype))
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
      if (is_index2t(value))
      {
        index2t &idx = to_index2t(value);
        idx.index = offset;
        exprt tmp_idx = migrate_expr_back(value);
        bounds_check(to_index_expr(tmp_idx), tmp_guard);
      }
      else if (!is_constant_int2t(offset) ||
               !to_constant_int2t(offset).constant_value.is_zero())
      {
        if(!options.get_bool_option("no-pointer-check"))
        {
          expr2tc zero = expr2tc(new constant_int2t(offset->type, BigInt(0)));
          expr2tc offs_is_not_zero = expr2tc(new notequal2t(offset, zero));

          guardt tmp_guard2(guard);
          exprt tmp_not_zero = migrate_expr_back(offs_is_not_zero);
          tmp_guard2.move(tmp_not_zero);

          dereference_callback.dereference_failure(
            "pointer dereference",
            "offset not zero (non-array-object)", tmp_guard2);
        }
      }
    }
  }
}

/*******************************************************************\

Function: dereferencet::valid_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereferencet::valid_check(
  const exprt &object,
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

  const exprt &symbol=get_symbol(object);

  if(symbol.id()=="string-constant")
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
  else if(symbol.is_nil() ||
          symbol.invalid_object())
  {
    // always "valid", shut up
    return;
  }
  else if(symbol.id()=="symbol")
  {
    const irep_idt identifier=symbol.identifier();

    if(dereference_callback.is_valid_object(identifier))
      return; // always ok
  }
}

/*******************************************************************\

Function: dereferencet::bounds_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereferencet::bounds_check(
  const index_exprt &expr,
  const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  const typet &array_type=ns.follow(expr.op0().type());

  if(!array_type.is_array())
    throw "bounds check expected array type";

  std::string name=array_name(ns, expr.array());

  {
    mp_integer i;
    if(!to_integer(expr.index(), i) &&
       i>=0)
    {
    }
    else
    {
      exprt zero=gen_zero(expr.index().type());

      if(zero.is_nil())
        throw "no zero constant of index type "+
          expr.index().type().to_string();

      binary_relation_exprt
        inequality(expr.index(), "<", zero);

      guardt tmp_guard(guard);
      tmp_guard.move(inequality);
      dereference_callback.dereference_failure(
        "array bounds",
        "`"+name+"' lower bound", tmp_guard);
    }
  }

  exprt size_expr=
    to_array_type(array_type).size();

  if (expr.op0().id() == "index")
  {
	std::string val1, val2, tot;
	int total;
    std::stringstream s;

	const typet array_type2=ns.follow(expr.op0().operands()[0].type());
	const exprt &size_expr2=to_array_type(array_type2).size();

	val1 = integer2string(binary2integer(size_expr.value().as_string(), true),10);
	val2 = integer2string(binary2integer(size_expr2.value().as_string(), true),10);
    total = atoi(val1.c_str())*atoi(val2.c_str());

    s << total;
    unsigned width;
    width = atoi(size_expr.type().width().as_string().c_str());
    constant_exprt value_expr(size_expr.type());
    value_expr.set_value(integer2binary(string2integer(s.str()),width));
    size_expr.swap(value_expr);
  }

  if(size_expr.id()!="infinity")
  {
    if(size_expr.is_nil())
      throw "index array operand of wrong type";

    binary_relation_exprt inequality(expr.index(), ">=", size_expr);

    if(c_implicit_typecast(
      inequality.op0(),
      inequality.op1().type(),
      ns))
      throw "index address of wrong type";

    guardt tmp_guard(guard);
    tmp_guard.move(inequality);

    dereference_callback.dereference_failure(
      "array bounds",
      "`"+name+"' upper bound", tmp_guard);
  }
}

/*******************************************************************\

Function: dereferencet::memory_model

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}

static bool is_a_bv_type(const typet &type)
{
  return type.id()=="unsignedbv" ||
         type.id()=="signedbv" ||
         type.id()=="bv" ||
         type.id()=="fixedbv" ||
         type.id()=="floatbv";
}

bool dereferencet::memory_model(
  exprt &value,
  const typet &to_type,
  const guardt &guard,
  exprt &new_offset)
{
  // we will allow more or less arbitrary pointer type cast

  const typet from_type=value.type();

  // first, check if it's really just a conversion

  if(is_a_bv_type(from_type) &&
     is_a_bv_type(to_type) &&
     bv_width(from_type)==bv_width(to_type))
    return memory_model_conversion(value, to_type, guard, new_offset);

  // otherwise, we will stich it together from bytes

  return memory_model_bytes(value, to_type, guard, new_offset);
}

/*******************************************************************\

Function: dereferencet::memory_model_conversion

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool dereferencet::memory_model_conversion(
  exprt &value,
  const typet &to_type,
  const guardt &guard,
  exprt &new_offset)
{
  const typet from_type=value.type();

  // avoid semantic conversion in case of
  // cast to float
  if(from_type.id()!="bv" &&
     (to_type.id()=="fixedbv" || to_type.id()=="floatbv"))
  {
    value.make_typecast(bv_typet(bv_width(from_type)));
    value.make_typecast(to_type);
  }
  else
  {
    // only doing type conversion
    // just do the typecast
    value.make_typecast(to_type);
  }

  // also assert that offset is zero

  if(!options.get_bool_option("no-pointer-check"))
  {
    equality_exprt offset_not_zero(new_offset, gen_zero(new_offset.type()));
    offset_not_zero.make_not();

    guardt tmp_guard(guard);
    tmp_guard.move(offset_not_zero);
    dereference_callback.dereference_failure(
      "word bounds",
      "offset not zero", tmp_guard);
  }

  return true;
}

/*******************************************************************\

Function: dereferencet::memory_model_bytes

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool dereferencet::memory_model_bytes(
  exprt &value,
  const typet &to_type,
  const guardt &guard,
  exprt &new_offset)
{
  const typet from_type=value.type();

  // we won't try to convert to/from code
  if(from_type.is_code() || to_type.is_code())
    return false;

  // won't do this without a committment to an endianess
  if(config.ansi_c.endianess==configt::ansi_ct::NO_ENDIANESS)
    return false;

  // But anything else we will try!

  // We allow reading more or less anything as bit-vector.
  if(to_type.id()=="bv" ||
     to_type.id()=="unsignedbv" ||
     to_type.id()=="signedbv")
  {
    const char *byte_extract_id=NULL;

    switch(config.ansi_c.endianess)
    {
    case configt::ansi_ct::IS_LITTLE_ENDIAN:
      byte_extract_id="byte_extract_little_endian";
      break;

    case configt::ansi_ct::IS_BIG_ENDIAN:
      byte_extract_id="byte_extract_big_endian";
      break;

    default:
      assert(false);
    }

    exprt byte_extract(byte_extract_id, to_type);
    byte_extract.copy_to_operands(value, new_offset);
    value=byte_extract;

    if(!new_offset.is_zero())
    {
      if(!options.get_bool_option("no-pointer-check"))
      {
        exprt bound=exprt("width", new_offset.type());
        bound.copy_to_operands(value.op0());

        binary_relation_exprt
          offset_upper_bound(new_offset, ">=", bound);

        guardt tmp_guard(guard);
        tmp_guard.move(offset_upper_bound);
        dereference_callback.dereference_failure(
          "word bounds",
          "word offset upper bound", tmp_guard);
      }

      if(!options.get_bool_option("no-pointer-check"))
      {
        binary_relation_exprt
          offset_lower_bound(new_offset, "<",
                             gen_zero(new_offset.type()));

        guardt tmp_guard(guard);
        tmp_guard.move(offset_lower_bound);
        dereference_callback.dereference_failure(
          "word bounds",
          "word offset lower bound", tmp_guard);
      }
    }

    return true;
  }

  return false;
}

