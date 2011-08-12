/*******************************************************************\

Module: Conversion of sizeof Expressions

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <config.h>
#include <arith_tools.h>

#include "c_sizeof.h"
#include "c_typecast.h"
#include "c_types.h"

/*******************************************************************\

Function: c_sizeoft::sizeof_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_sizeoft::sizeof_rec(const typet &type)
{
  exprt dest;

  if(type.is_signedbv() ||
     type.id()=="unsignedbv" ||
     type.is_floatbv() ||
     type.is_fixedbv())
  {
    unsigned bits=atoi(type.width().c_str());
    unsigned bytes=bits/8;
    if((bits%8)!=0) bytes++;
    dest=from_integer(bits/8, uint_type());
  }
  else if(type.id()=="c_enum" ||
          type.id()=="incomplete_c_enum")
  {
    dest=from_integer(config.ansi_c.int_width/8, uint_type());
  }
  else if(type.is_pointer())
  {
    if(type.reference())
      return sizeof_rec(type.subtype());

    unsigned bits=config.ansi_c.pointer_width;
    dest=from_integer(bits/8, uint_type());
    if((bits%8)!=0) dest.make_nil();
  }
  else if(type.is_bool())
  {
    dest=from_integer(1, uint_type());
  }
  else if(type.is_array())
  {
    const exprt &size_expr=
      static_cast<const exprt &>(type.size_irep());

    exprt tmp_dest(sizeof_rec(type.subtype()));

    if(tmp_dest.is_nil())
      return tmp_dest;

    mp_integer a, b;

    if(!to_integer(tmp_dest, a) &&
       !to_integer(size_expr, b))
    {
      dest=from_integer(a*b, uint_type());
    }
    else
    {
      dest.id("*");
      dest.type()=uint_type();
      dest.copy_to_operands(size_expr);
      dest.move_to_operands(tmp_dest);
      c_implicit_typecast(dest.op0(), dest.type(), ns);
      c_implicit_typecast(dest.op1(), dest.type(), ns);
    }
  }
  else if(type.is_struct())
  {
    const irept::subt &components=
      type.components().get_sub();

    mp_integer sum=0;

    forall_irep(it, components)
    {
      const typet &sub_type=it->type();

      if(sub_type.is_code())
      {
      }
      else
      {
        exprt tmp(sizeof_rec(sub_type));

        if(tmp.is_nil())
          return tmp;

        mp_integer tmp_int;
        if(to_integer(tmp, tmp_int))
          return static_cast<const exprt &>(get_nil_irep());

        sum+=tmp_int;
      }
    }

    dest=from_integer(sum, uint_type());
  }
  else if(type.id()=="union")
  {
    const irept::subt &components=
      type.components().get_sub();

    mp_integer max_size=0;

    forall_irep(it, components)
    {
      const typet &sub_type=it->type();

      if(sub_type.is_code())
      {
      }
      else
      {
        exprt tmp(sizeof_rec(sub_type));

        if(tmp.is_nil())
          return tmp;

        mp_integer tmp_int;

        if(to_integer(tmp, tmp_int))
          return static_cast<const exprt &>(get_nil_irep());

        if(tmp_int>max_size) max_size=tmp_int;
      }
    }

    dest=from_integer(max_size, uint_type());
  }
  else if(type.id()=="symbol")
  {
    return sizeof_rec(ns.follow(type));
  }
  else
    dest.make_nil();

  return dest;
}
