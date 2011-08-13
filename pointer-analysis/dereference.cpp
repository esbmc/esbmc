/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

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
#include "pointer_offset_sum.h"

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

  if(expr.is_dereference() ||
     expr.is_implicit_dereference() ||
     (expr.is_index() && expr.operands().size()==2 &&
      expr.op0().type().is_pointer()))
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
  if(expr.is_member() || expr.is_index())
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
  if(!dest.type().is_pointer())
    throw "dereference expected pointer type, but got "+
          dest.type().pretty();

  // save the dest for later, dest might be destroyed
  const exprt deref_expr(dest);

  // type of the object
  const typet &type=deref_expr.type().subtype();

  #if 0
  std::cout << "DEREF: " << dest.pretty() << std::endl;
  #endif

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

    build_reference_to(
      *it, mode, dest, type,
      new_value, pointer_guard, guard);

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
  if(!dest.type().is_pointer())
    throw "dereference expected pointer type, but got "+
          dest.type().pretty();

  #if 0
  std::cout << "ADD CHECK: " << dest.pretty() << std::endl;
  #endif

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
      exprt new_value, pointer_guard;

      build_reference_to(
        *it, mode, dest, type,
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

  if(dereference_type.is_empty())
    return true; // always ok

  if(base_type_eq(object_type, dereference_type, ns))
    return true; // ok, they just match

  // check for struct prefixes

  typet ot_base(object_type),
        dt_base(dereference_type);

  base_type(ot_base, ns);
  base_type(dt_base, ns);

  if(ot_base.is_struct() &&
     dt_base.is_struct())
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
  const exprt &what,
  const modet mode,
  const exprt &deref_expr,
  const typet &type,
  exprt &value,
  exprt &pointer_guard,
  const guardt &guard)
{
  value.make_nil();
  pointer_guard.make_false();

  if(what.is_unknown() ||
     what.is_invalid())
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is an invalid pointer

      exprt invalid_pointer_expr("invalid-pointer", typet("bool"));
      invalid_pointer_expr.copy_to_operands(deref_expr);

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

  if(what.id()!="object_descriptor")
    throw "unknown points-to: "+what.id_string();

  const object_descriptor_exprt &o=to_object_descriptor_expr(what);

  const exprt &root_object=o.root_object();
  const exprt &object=o.object();

  if(root_object.is_null_object())
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      constant_exprt pointer(typet("pointer"));
      pointer.type().subtype()=type;
      pointer.set_value("NULL");

      exprt pointer_guard("same-object", typet("bool"));
      pointer_guard.copy_to_operands(deref_expr, pointer);

      guardt tmp_guard(guard);
      tmp_guard.add(pointer_guard);

      dereference_callback.dereference_failure(
        "pointer dereference",
        "NULL pointer", tmp_guard);
    }
  }
  else if(root_object.is_dynamic_object())
  {
    const dynamic_object_exprt &dynamic_object=
      to_dynamic_object_expr(root_object);

    value=exprt("dereference", type);
    value.copy_to_operands(deref_expr);

    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is a dynamic object

      exprt is_dynamic_object_expr("is_dynamic_object", typet("bool"));
      is_dynamic_object_expr.copy_to_operands(deref_expr);

      if(!dynamic_object.valid().is_true())
      {
        // check if it is still alive
        exprt valid_expr("valid_object", typet("bool"));
        valid_expr.copy_to_operands(deref_expr);
        valid_expr.make_not();

        guardt tmp_guard(guard);
        tmp_guard.add(is_dynamic_object_expr);
        tmp_guard.move(valid_expr);
        dereference_callback.dereference_failure(
          "pointer dereference",
          "invalidated dynamic object",
          tmp_guard);
      }

#if 1
      if(!options.get_bool_option("no-bounds-check") &&
         !o.offset().is_zero())
      {
        {
          // check lower bound
          exprt zero=gen_zero(index_type());
          assert(zero.is_not_nil());

          exprt object_offset=exprt("pointer_offset", index_type());
          object_offset.copy_to_operands(deref_expr);

          binary_relation_exprt
            inequality(object_offset, "<", zero);

          guardt tmp_guard(guard);
          tmp_guard.add(is_dynamic_object_expr);
          tmp_guard.move(inequality);
          dereference_callback.dereference_failure(
            "pointer dereference",
            "dynamic object lower bound", tmp_guard);
        }

        {
          // check upper bound
          //nec: ex37.c
          exprt size_expr=exprt("dynamic_size", int_type()/*uint_type()*/);
          size_expr.copy_to_operands(deref_expr);

          exprt object_offset=exprt("pointer_offset", index_type());
          object_offset.copy_to_operands(deref_expr);
          object_offset.make_typecast(int_type()/*uint_type()*/);

          binary_relation_exprt
            inequality(size_expr, "<=", object_offset);

          guardt tmp_guard(guard);
          tmp_guard.add(is_dynamic_object_expr);
          tmp_guard.move(inequality);

          //std::cout << "tmp_guard: " << tmp_guard.as_expr().pretty() << std::endl;

          dereference_callback.dereference_failure(
            "pointer dereference",
            "dynamic object upper bound", tmp_guard);
        }
      }
#endif
#if 0
      exprt tmp_object(object);

      // check type
      if(!dereference_type_compare(tmp_object, type))
      {
        exprt type_expr=exprt("pointer_object_has_type", typet("bool"));
        type_expr.copy_to_operands(deref_expr);
        type_expr.object_type(type);
        type_expr.make_not();

        guardt tmp_guard(guard);
        tmp_guard.add(is_dynamic_object_expr);
        tmp_guard.move(type_expr);

        std::string msg="wrong object type (got `";
        msg+=from_type(ns, "", object.type());
        msg+="', expected `";
        msg+=from_type(ns, "", type);
        msg+="')";

        dereference_callback.dereference_failure(
          "pointer dereference",
          msg, tmp_guard);
      }
#endif
    }
  }
  else
  {
    value=object;

    exprt object_pointer("address_of", pointer_typet());
    object_pointer.type().subtype()=object.type();
    object_pointer.copy_to_operands(object);
    //std::cout << "deref_expr.pretty(): " << deref_expr.pretty() << std::endl;
    pointer_guard=exprt("same-object", typet("bool"));
    pointer_guard.copy_to_operands(deref_expr, object_pointer);

    guardt tmp_guard(guard);
    tmp_guard.add(pointer_guard);

    //std::cout << "tmp_guard.as_expr().pretty(): " << tmp_guard.as_expr().pretty() << std::endl;
    valid_check(object, tmp_guard, mode);

    exprt offset;

    //std::cout << "o.offset().is_constant(): " << o.offset().is_constant() << std::endl;

    if(o.offset().is_constant())
      offset=o.offset();
    else
    {
      exprt pointer_offset=exprt("pointer_offset", index_type());
      pointer_offset.copy_to_operands(deref_expr);

      exprt base=exprt("pointer_offset", index_type());
      base.copy_to_operands(object_pointer);

      // need to subtract base address
      offset=exprt("-", index_type());
      offset.move_to_operands(pointer_offset, base);
      //std::cout << "offset.pretty(): " << offset.pretty() << std::endl;
    }
    //std::cout << "!dereference_type_compare(value, type): " << !dereference_type_compare(value, type) << std::endl;
    if(!dereference_type_compare(value, type))
    {
      if(memory_model(value, type, tmp_guard, offset))
      {
        // ok
      }
      else
      {
        if(!options.get_bool_option("no-pointer-check"))
        {
          //nec: ex29
          if (value.type().subtype().is_empty() ||
        		  type.subtype().is_empty())
            return;
          std::string msg="memory model not applicable (got `";
          msg+=from_type(ns, "", value.type());
          msg+="', expected `";
          msg+=from_type(ns, "", type);
          msg+="')";

          dereference_callback.dereference_failure(
            "pointer dereference",
            msg, tmp_guard);
        }

        value.make_nil();
        return; // give up, no way that this is ok
      }
    }
    else
    {
      //std::cout << "value.id(): " << value.id() << std::endl;
      if(value.is_index())
      {
        index_exprt &index_expr=to_index_expr(value);
        index_expr.index()=offset;
        //std::cout << "value.pretty(): " << value.pretty() << std::endl;
        //std::cout << "index_expr.pretty(): " << index_expr.pretty() << std::endl;
        //std::cout << "tmp_guard.pretty(): " << tmp_guard.as_expr().pretty() << std::endl;
        //std::cout << "tmp_guard.as_expr().id(): " << tmp_guard.as_expr().id() << std::endl;
        bounds_check(index_expr, tmp_guard);
      }
      else if(!offset.is_zero())
      {
        if(!options.get_bool_option("no-pointer-check"))
        {
          equality_exprt
            offset_not_zero(offset, gen_zero(offset.type()));
          offset_not_zero.make_not();

          guardt tmp_guard2(guard);
          tmp_guard2.move(offset_not_zero);

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

  if(symbol.is_string_constant())
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
  else if(symbol.is_symbol())
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
  //std::cout << "expr.op0().pretty(): " << expr.op0().pretty() << std::endl;
  const typet &array_type=ns.follow(expr.op0().type());
  //std::cout << "array_type.pretty(): " << array_type.pretty() << std::endl;
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

//  const exprt &size_expr=
//    to_array_type(array_type).size();

  exprt size_expr=
    to_array_type(array_type).size();

  if (expr.op0().is_index())
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
    //std::cout << "tmp_guard.as_expr().pretty(): " << tmp_guard.as_expr().pretty() << std::endl;
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
  return type.is_unsignedbv() ||
         type.is_signedbv() ||
         type.is_bv() ||
         type.is_fixedbv() ||
         type.is_floatbv();
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
     (to_type.is_fixedbv() || to_type.is_floatbv()))
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
  if(to_type.is_bv() ||
     to_type.is_unsignedbv() ||
     to_type.is_signedbv())
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

