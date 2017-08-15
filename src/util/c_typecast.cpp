/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <util/base_type.h>
#include <util/c_qualifiers.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/irep2_utils.h>
#include <util/simplify_expr_class.h>
#include <util/std_expr.h>
#include <util/string2array.h>

// In this file, all functions and methods are replicated with irept and irep2
// copies, because it's unclear to me what the overall algorithm is, therefore
// I can't replicated it in another way for irep2. Too bad.

bool c_implicit_typecast(
  exprt &expr,
  const typet &dest_type,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast(expr, dest_type);
  return !c_typecast.errors.empty();
}

bool c_implicit_typecast(
  expr2tc &expr,
  const type2tc &dest_type,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast(expr, dest_type);
  return !c_typecast.errors.empty();
}

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

bool check_c_implicit_typecast(
  const type2tc &src_type,
  const type2tc &dest_type,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  // It seems that this expression is just a vehicle for the type to be
  // renamed.
  symbol2tc tmp(src_type, "shoes");
  c_typecast.implicit_typecast(tmp, dest_type);
  return !c_typecast.errors.empty();
}

bool c_implicit_typecast_arithmetic(
  exprt &expr1, exprt &expr2,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast_arithmetic(expr1, expr2);
  return !c_typecast.errors.empty();
}

bool c_implicit_typecast_arithmetic(
  expr2tc &expr1, expr2tc &expr2,
  const namespacet &ns)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast_arithmetic(expr1, expr2);
  return !c_typecast.errors.empty();
}

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

bool check_c_implicit_typecast(
  const type2tc &src_type,
  const type2tc &dest_type)
{
  // check qualifiers

  // irep2 doesn't have various of the things in the original copy of this
  // function, in particular cmt_constant and things like "natural" or
  // "integer", so they're deleted here.

  if(src_type==dest_type) return false;

  if (is_bool_type(src_type))
  {
    if (is_unsignedbv_type(dest_type)) return false;
    if (is_signedbv_type(dest_type)) return false;
    if (is_pointer_type(dest_type)) return false;
    if (is_fixedbv_type(dest_type)) return false;
  }
  else if (is_bv_type(src_type))
  {
    if (is_bool_type(dest_type)) return false;
    if (is_unsignedbv_type(dest_type)) return false;
    if (is_signedbv_type(dest_type)) return false;
    if (is_pointer_type(dest_type)) return false;
    if (is_fixedbv_type(dest_type)) return false;
  }
  else if (is_fixedbv_type(src_type))
  {
    if (is_bool_type(dest_type)) return false;
    if (is_unsignedbv_type(dest_type)) return false;
    if (is_signedbv_type(dest_type)) return false;
    if (is_fixedbv_type(dest_type)) return false;
  }
  else if (is_array_type(src_type) || is_pointer_type(src_type))
  {
    if (is_pointer_type(dest_type))
    {
      pointer_type2tc dest_ptr_type = dest_type;
      pointer_type2tc src_ptr_type = src_type;
      type2tc dest_subtype = dest_ptr_type->subtype;
      type2tc src_subtype = src_ptr_type->subtype;

      if (src_subtype==dest_subtype)
        return false;
      else if (is_empty_type(src_subtype) || // from void to anything
               is_empty_type(dest_subtype)) // to void from anything
        return false;
    }

    if (is_array_type(dest_type)) {
      type2tc src_subtype;
      if (is_pointer_type(src_type))
        src_subtype = to_pointer_type(src_type).subtype;
      else
        src_subtype = to_array_type(src_type).subtype;

      if (src_subtype == to_array_type(dest_type).subtype)
        return false;
    }

    if (is_bool_type(dest_type)) return false;
    if (is_unsignedbv_type(dest_type)) return false;
    if (is_signedbv_type(dest_type)) return false;
  }

  return true;
}

typet c_typecastt::follow_with_qualifiers(const typet &src_type)
{
  if(src_type.id()!="symbol") return src_type;

  c_qualifierst qualifiers(src_type);

  typet dest_type=ns.follow(src_type);
  qualifiers.write(dest_type);

  return dest_type;
}

type2tc
c_typecastt::follow_with_qualifiers(const type2tc &src_type)
{
  if (!is_symbol_type(src_type))
    return src_type;

  type2tc dest_type = ns.follow(src_type);
  // TODO: We ditch qualifiers during symex, which is the only place that
  // irep2 exists right now. If we ever move frontends over to new irep,
  // then this will become an issue, hence the following warning.
  return dest_type;
}

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

c_typecastt::c_typet c_typecastt::get_c_type(
  const type2tc &type)
{

  if (is_signedbv_type(type))
  {
    signedbv_type2tc signed_type = type;
    unsigned width = signed_type->width;

    if(width<=config.ansi_c.char_width)
      return CHAR;
    else if(width<=config.ansi_c.int_width)
      return INT;
    else if(width<=config.ansi_c.long_int_width)
      return LONG;
    else if(width<=config.ansi_c.long_long_int_width)
      return LONGLONG;
  }
  else if (is_unsignedbv_type(type))
  {
    unsignedbv_type2tc unsigned_type = type;
    unsigned width = unsigned_type->width;

    if(width<=config.ansi_c.char_width)
      return UCHAR;
    else if(width<=config.ansi_c.int_width)
      return UINT;
    else if(width<=config.ansi_c.long_int_width)
      return ULONG;
    else if(width<=config.ansi_c.long_long_int_width)
      return ULONGLONG;
  }
  else if (is_bool_type(type))
    return BOOL;
  else if (is_fixedbv_type(type))
  {
    fixedbv_type2tc fixedbv_type = type;
    unsigned width = fixedbv_type->width;
    if(width<=config.ansi_c.single_width)
      return SINGLE;
    else if(width<=config.ansi_c.double_width)
      return DOUBLE;
    else if(width<=config.ansi_c.long_double_width)
      return LONGDOUBLE;
  }
  else if (is_pointer_type(type))
  {
    pointer_type2tc ptr_type = type;
    if (is_empty_type(ptr_type->subtype))
      return VOIDPTR;
    else
      return PTR;
  }
  else if (is_array_type(type))
  {
    return PTR;
  }
  else if (is_symbol_type(type))
    return get_c_type(ns.follow(type));

  return OTHER;
}

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

void c_typecastt::implicit_typecast_arithmetic(
  expr2tc &expr,
  c_typet c_type)
{
  type2tc new_type;

  const type2tc &expr_type = ns.follow(expr->type);

  switch(c_type)
  {
  case PTR:
    if (is_array_type(expr_type))
    {
      array_type2tc arr_type = expr_type;
      new_type = pointer_type2tc(arr_type->subtype);
      break;
    }
    return;

  case BOOL:       new_type = get_bool_type(); break;
  case CHAR:       assert(false); // should always be promoted
  case UCHAR:      assert(false); // should always be promoted
                   abort();
  case INT:        new_type=int_type2(); break;
  case UINT:       new_type=uint_type2(); break;
  case LONG:       new_type=long_int_type2(); break;
  case ULONG:      new_type=long_uint_type2(); break;
  case LONGLONG:   new_type=long_long_int_type2(); break;
  case ULONGLONG:  new_type=long_long_uint_type2(); break;
  case SINGLE:     new_type=float_type2(); break;
  case DOUBLE:     new_type=double_type2(); break;
  case LONGDOUBLE: new_type=long_double_type2(); break;
  default: return;
  }

  if(new_type!=expr_type)
  {
    if (is_pointer_type(new_type) && is_array_type(expr_type))
    {
      array_type2tc arr_type = expr_type;
      pointer_type2tc ptr_type = new_type;
      index2tc index_expr(arr_type->subtype, expr, gen_zero(index_type2()));
      address_of2tc addrof(ptr_type->subtype, index_expr);
      expr = addrof;
    }
    else
      do_typecast(expr, new_type);
  }
}

void c_typecastt::implicit_typecast_arithmetic(exprt &expr)
{
  c_typet c_type=get_c_type(expr.type());
  c_type=std::max(c_type, INT); // minimum promotion
  implicit_typecast_arithmetic(expr, c_type);
}

void c_typecastt::implicit_typecast_arithmetic(expr2tc &expr)
{
  c_typet c_type = get_c_type(expr->type);
  c_type = std::max(c_type, INT); // minimum promotion
  implicit_typecast_arithmetic(expr, c_type);
}

void c_typecastt::implicit_typecast(
  exprt &expr,
  const typet &type)
{
  typet src_type=follow_with_qualifiers(expr.type()),
        dest_type=follow_with_qualifiers(type);

  implicit_typecast_followed(expr, src_type, dest_type);
}

void c_typecastt::implicit_typecast(
  expr2tc &expr,
  const type2tc &type)
{
  type2tc src_type=follow_with_qualifiers(expr->type),
          dest_type=follow_with_qualifiers(type);

  implicit_typecast_followed(expr, src_type, dest_type);
}

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
        warnings.emplace_back("incompatible pointer types");

      // check qualifiers

      if(src_type.subtype().cmt_constant() &&
         !dest_type.subtype().cmt_constant())
        warnings.emplace_back("disregarding const");

      if(src_type.subtype().cmt_volatile() &&
         !dest_type.subtype().cmt_volatile())
        warnings.emplace_back("disregarding volatile");

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
    errors.emplace_back("implicit conversion not permitted");
  else if(src_type!=dest_type)
    do_typecast(expr, dest_type);
}

void c_typecastt::implicit_typecast_followed(
  expr2tc &expr,
  const type2tc &src_type,
  const type2tc &dest_type)
{
  if (is_pointer_type(dest_type))
  {
    pointer_type2tc dest_ptr_type = dest_type;
    // special case: 0 == NULL

    if (is_constant_int2t(expr) &&
        to_constant_int2t(expr).value.to_long() == 0 && (
       is_unsignedbv_type(src_type) || is_signedbv_type(src_type)))
    {
      expr = symbol2tc(dest_type, "NULL");
      return; // ok
    }

    if (is_pointer_type(src_type) || is_array_type(src_type))
    {
      // we are quite generous about pointers
      type2tc src_subtype;
      if (is_pointer_type(src_type))
        src_subtype = to_pointer_type(src_type).subtype;
      else
        src_subtype = to_array_type(src_type).subtype;

      const type2tc &src_sub = ns.follow(src_subtype);
      const type2tc &dest_sub = ns.follow(dest_ptr_type->subtype);

      if (is_empty_type(src_sub) || is_empty_type(dest_sub))
      {
        // from/to void is always good
      }
      else if(base_type_eq(dest_ptr_type->subtype, src_subtype, ns))
      {
      }
      else if (is_code_type(src_sub) && is_code_type(dest_sub))
      {
        // very generous:
        // between any two function pointers it's ok
      }
      else if (is_bv_type(src_sub) && is_bv_type(dest_sub))
      {
        // also generous: between any to scalar types it's ok
      }
      else
        warnings.push_back("incompatible pointer types");

      if (src_type==dest_type)
      {
        expr.get()->type = src_type; // because of qualifiers
      }
      else
        do_typecast(expr, dest_type);

      return; // ok
    }
  }

  if(check_c_implicit_typecast(src_type, dest_type))
    errors.push_back("implicit conversion not permitted");
  else if(src_type!=dest_type)
    do_typecast(expr, dest_type);
}

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

void c_typecastt::implicit_typecast_arithmetic(
  expr2tc &expr1,
  expr2tc &expr2)
{
  const type2tc &type1 = ns.follow(expr1->type);
  const type2tc &type2 = ns.follow(expr2->type);

  c_typet c_type1=get_c_type(type1),
          c_type2=get_c_type(type2);

  c_typet max_type=std::max(c_type1, c_type2);
  max_type=std::max(max_type, INT); // minimum promotion

  implicit_typecast_arithmetic(expr1, max_type);
  implicit_typecast_arithmetic(expr2, max_type);

  if(max_type==PTR)
  {
    if(c_type1==VOIDPTR)
      do_typecast(expr1, expr2->type);

    if(c_type2==VOIDPTR)
      do_typecast(expr2, expr1->type);
  }
}

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

void c_typecastt::do_typecast(expr2tc &dest, const type2tc &type)
{
  // special case: array -> pointer is actually
  // something like address_of

  const type2tc &dest_type = ns.follow(dest->type);

  if (is_array_type(dest_type))
  {
    array_type2tc arr_type = dest_type;
    index2tc index(arr_type->subtype, dest, gen_zero(index_type2()));
    address_of2tc tmp(arr_type->subtype, index);
    dest = tmp;
    if (ns.follow(dest->type) != ns.follow(type))
      dest = typecast2tc(type, dest);
    return;
  }

  if(dest_type!=type)
  {
    dest = typecast2tc(type, dest);

#if 0
    // jmorse - ???????
    if (dest.op0().is_constant())
    {
      // preserve #c_sizeof_type -- don't make it a reference!
      const irept c_sizeof_type=
        dest.op0().c_sizeof_type();

      simplify_exprt simplify_expr;
      simplify_expr.simplify_typecast(dest, simplify_exprt::NORMAL);

      if(c_sizeof_type.is_not_nil())
        dest.cmt_c_sizeof_type(c_sizeof_type);
    }
#endif
  }
}
