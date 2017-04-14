/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/c_qualifiers.h>
#include <ansi-c/c_typecast.h>
#include <ansi-c/type2name.h>
#include <cassert>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/simplify_expr_class.h>
#include <util/std_expr.h>
#include <util/string2array.h>

/*******************************************************************\

Function: c_implicit_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_implicit_typecast(
  exprt &expr,
  const typet &dest_type,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast(expr, dest_type);
  return !c_typecast.errors.empty();
}

/*******************************************************************\

Function: check_c_implicit_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool check_c_implicit_typecast(
  const typet &src_type,
  const typet &dest_type,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  exprt tmp;
  tmp.type()=src_type;
  c_typecast.implicit_typecast(tmp, dest_type);
  return !c_typecast.errors.empty();
}

/*******************************************************************\

Function: c_implicit_typecast_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_implicit_typecast_arithmetic(
  exprt &expr1, exprt &expr2,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast_arithmetic(expr1, expr2);
  return !c_typecast.errors.empty();
}

/*******************************************************************\

Function: check_c_implicit_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool check_c_implicit_typecast(
  const typet &src_type,
  const typet &dest_type)
{
  // check qualifiers

  if(src_type.id()=="pointer" && dest_type.id()=="pointer" &&
     src_type.subtype().cmt_constant() &&
     !dest_type.subtype().cmt_constant())
    return true;

  if(src_type==dest_type) return false;

  const std::string &src_type_id=src_type.id_string();

  if(src_type_id=="bool")
  {
    if(dest_type.id()=="unsignedbv") return false;
    if(dest_type.id()=="signedbv") return false;
    if(dest_type.id()=="pointer") return false;
    if(dest_type.id()=="floatbv") return false;
    if(dest_type.id()=="fixedbv") return false;
    if(dest_type.id()=="c_enum") return false;
  }
  else if(src_type_id=="unsignedbv" ||
          src_type_id=="signedbv" ||
          src_type_id=="c_enum" ||
          src_type_id=="incomplete_c_enum")
  {
    if(dest_type.id()=="unsignedbv") return false;
    if(dest_type.is_bool()) return false;
    if(dest_type.id()=="signedbv") return false;
    if(dest_type.id()=="floatbv") return false;
    if(dest_type.id()=="fixedbv") return false;
    if(dest_type.id()=="pointer") return false;
    if(dest_type.id()=="c_enum") return false;
    if(dest_type.id()=="incomplete_c_enum") return false;
  }
  else if(src_type_id=="floatbv" ||
          src_type_id=="fixedbv")
  {
    if(dest_type.is_bool()) return false;
    if(dest_type.id()=="signedbv") return false;
    if(dest_type.id()=="unsignedbv") return false;
    if(dest_type.id()=="floatbv") return false;
    if(dest_type.id()=="fixedbv") return false;
  }
  else if(src_type_id=="array" ||
          src_type_id=="incomplete_array" ||
          src_type_id=="pointer")
  {
    if(dest_type.id()=="pointer")
    {
      const irept &dest_subtype=dest_type.subtype();
      const irept &src_subtype =src_type.subtype();

      if(src_subtype==dest_subtype)
        return false;
      else if(src_subtype.id()=="empty" || // from void to anything
              dest_subtype.id()=="empty")  // to void from anything
        return false;
    }

    if((dest_type.is_array() ||
        dest_type.id()=="incomplete_array") &&
       (src_type.subtype()==dest_type.subtype())) return false;

    if(dest_type.is_bool()) return false;
    if(dest_type.id()=="unsignedbv") return false;
    if(dest_type.id()=="signedbv") return false;
  }

  return true;
}

/*******************************************************************\

Function: c_typecastt::follow_with_qualifiers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

typet c_typecastt::follow_with_qualifiers(const typet &src_type)
{
  if(src_type.id()!="symbol") return src_type;

  c_qualifierst qualifiers(src_type);

  typet dest_type=ns.follow(src_type);
  qualifiers.write(dest_type);

  return dest_type;
}

/*******************************************************************\

Function: c_typecastt::get_c_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

c_typecastt::c_typet c_typecastt::get_c_type(
  const typet &type)
{
  unsigned width=atoi(type.width().c_str());

  if(type.id()=="signedbv")
  {
    if(width<=config.ansi_c.char_width)
      return CHAR;
    else if(width<=config.ansi_c.int_width)
      return INT;
    else if(width<=config.ansi_c.long_int_width)
      return LONG;
    else if(width<=config.ansi_c.long_long_int_width)
      return LONGLONG;
  }
  else if(type.id()=="unsignedbv")
  {
    if(width<=config.ansi_c.char_width)
      return UCHAR;
    else if(width<=config.ansi_c.int_width)
      return UINT;
    else if(width<=config.ansi_c.long_int_width)
      return ULONG;
    else if(width<=config.ansi_c.long_long_int_width)
      return ULONGLONG;
  }
  else if(type.is_bool())
    return BOOL;
  else if(type.id()=="floatbv" ||
          type.id()=="fixedbv")
  {
    if(width<=config.ansi_c.single_width)
      return SINGLE;
    else if(width<=config.ansi_c.double_width)
      return DOUBLE;
    else if(width<=config.ansi_c.long_double_width)
      return LONGDOUBLE;
  }
  else if(type.id()=="pointer")
  {
    if(type.subtype().id()=="empty")
      return VOIDPTR;
    else
      return PTR;
  }
  else if(type.is_array() ||
          type.id()=="incomplete_array")
  {
    return PTR;
  }
  else if(type.id()=="c_enum" ||
          type.id()=="incomplete_c_enum")
  {
    return INT;
  }
  else if(type.id()=="symbol")
    return get_c_type(ns.follow(type));

  return OTHER;
}

/*******************************************************************\

Function: c_typecastt::implicit_typecast_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::implicit_typecast_arithmetic(
  exprt &expr,
  c_typet c_type)
{
  typet new_type;

  const typet &expr_type=ns.follow(expr.type());

  switch(c_type)
  {
  case PTR:
    if(expr_type.is_array())
    {
      new_type.id("pointer");
      new_type.subtype()=expr_type.subtype();
      break;
    }
    return;

  case BOOL:       new_type=bool_type(); break;
  case CHAR:       assert(false); // should always be promoted
  case UCHAR:      assert(false); // should always be promoted
  case INT:        new_type=int_type(); break;
  case UINT:       new_type=uint_type(); break;
  case LONG:       new_type=long_int_type(); break;
  case ULONG:      new_type=long_uint_type(); break;
  case LONGLONG:   new_type=long_long_int_type(); break;
  case ULONGLONG:  new_type=long_long_uint_type(); break;
  case SINGLE:     new_type=float_type(); break;
  case DOUBLE:     new_type=double_type(); break;
  case LONGDOUBLE: new_type=long_double_type(); break;
  default: return;
  }

  if(new_type!=expr_type)
    do_typecast(expr, new_type);
}

/*******************************************************************\

Function: c_typecastt::implicit_typecast_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::implicit_typecast_arithmetic(exprt &expr)
{
  c_typet c_type=get_c_type(expr.type());
  c_type=std::max(c_type, INT); // minimum promotion
  implicit_typecast_arithmetic(expr, c_type);
}

/*******************************************************************\

Function: c_typecastt::implicit_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::implicit_typecast(
  exprt &expr,
  const typet &type)
{
  typet src_type=follow_with_qualifiers(expr.type()),
        dest_type=follow_with_qualifiers(type);

  implicit_typecast_followed(expr, src_type, dest_type);
}

/*******************************************************************\

Function: c_typecastt::implicit_typecast_followed

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::implicit_typecast_followed(
  exprt &expr,
  const typet &src_type,
  const typet &dest_type)
{
  if(dest_type.id()=="pointer")
  {
    // special case: 0 == NULL

    if(expr.is_zero() && (
       src_type.id()=="unsignedbv" ||
       src_type.id()=="signedbv"))
    {
      expr=exprt("constant", dest_type);
      expr.value("NULL");
      return; // ok
    }

    if(src_type.id()=="pointer" ||
       src_type.is_array() ||
       src_type.id()=="incomplete_array")
    {
      // we are quite generous about pointers

      const typet &src_sub=ns.follow(src_type.subtype());
      const typet &dest_sub=ns.follow(dest_type.subtype());

      if(src_sub.id()=="empty" ||
         dest_sub.id()=="empty")
      {
        // from/to void is always good
      }
      else if(base_type_eq(
        dest_type.subtype(), src_type.subtype(), ns))
      {
      }
      else if(src_sub.is_code() &&
              dest_sub.is_code())
      {
        // very generous:
        // between any two function pointers it's ok
      }
      else if(is_number(src_sub) && is_number(dest_sub))
      {
        // also generous: between any to scalar types it's ok
      }
      else
        warnings.push_back("incompatible pointer types");

      // check qualifiers

      if(src_type.subtype().cmt_constant() &&
         !dest_type.subtype().cmt_constant())
        warnings.push_back("disregarding const");

      if(src_type.subtype().cmt_volatile() &&
         !dest_type.subtype().cmt_volatile())
        warnings.push_back("disregarding volatile");

      if(src_type==dest_type)
      {
        expr.type()=src_type; // because of qualifiers
      }
      else
        do_typecast(expr, dest_type);

      return; // ok
    }
  }
  else if(dest_type.id()=="array")
  {
    if(expr.id() == "string-constant")
    {
      expr.type() = dest_type;

      exprt dest;
      string2array(expr, dest);
      expr.swap(dest);

      return;
    }
  }

  if(check_c_implicit_typecast(src_type, dest_type))
    errors.push_back("implicit conversion not permitted");
  else if(src_type!=dest_type)
    do_typecast(expr, dest_type);
}

/*******************************************************************\

Function: c_typecastt::implicit_typecast_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::implicit_typecast_arithmetic(
  exprt &expr1,
  exprt &expr2)
{
  const typet &type1=ns.follow(expr1.type());
  const typet &type2=ns.follow(expr2.type());

  c_typet c_type1=get_c_type(type1),
          c_type2=get_c_type(type2);

  c_typet max_type=std::max(c_type1, c_type2);
  max_type=std::max(max_type, INT); // minimum promotion

  implicit_typecast_arithmetic(expr1, max_type);
  implicit_typecast_arithmetic(expr2, max_type);

  if(max_type==PTR)
  {
    if(c_type1==VOIDPTR)
      do_typecast(expr1, expr2.type());

    if(c_type2==VOIDPTR)
      do_typecast(expr2, expr1.type());
  }
}

/*******************************************************************\

Function: c_typecastt::do_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecastt::do_typecast(exprt &dest, const typet &type)
{
  // special case: array -> pointer is actually
  // something like address_of

  const typet &dest_type=ns.follow(dest.type());

  if(dest_type.is_array())
  {
    index_exprt index;
    index.array()=dest;
    index.index()=gen_zero(index_type());
    index.type()=dest_type.subtype();
    dest=gen_address_of(index);
    if(ns.follow(dest.type()) != ns.follow(type))
      dest.make_typecast(type);
    return;
  }

  if(dest_type!=type)
  {
    dest.make_typecast(type);

    if(dest.op0().is_constant())
    {
      // preserve #c_sizeof_type -- don't make it a reference!
      const irept c_sizeof_type=
        dest.op0().c_sizeof_type();

      simplify_exprt simplify_expr;
      simplify_expr.simplify_typecast(dest);

      if(c_sizeof_type.is_not_nil())
        dest.c_sizeof_type(c_sizeof_type);
    }
  }
}
