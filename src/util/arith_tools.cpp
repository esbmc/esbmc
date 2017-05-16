/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <util/arith_tools.h>
#include <util/bitvector.h>

bool to_integer(const exprt &expr, mp_integer &int_value)
{
  if(!expr.is_constant()) return true;

  const std::string &value=expr.value().as_string();
  const irep_idt &type_id=expr.type().id();

  if(type_id=="pointer")
  {
    if(value=="NULL")
    {
      int_value=0;
      return false;
    }
  }
  else if(type_id=="c_enum"  ||
          type_id=="symbol")
  {
    int_value=string2integer(value);
    return false;
  }
  else if(type_id=="unsignedbv")
  {
    int_value=binary2integer(value, false);
    return false;
  }
  else if(type_id=="signedbv")
  {
    int_value=binary2integer(value, true);
    return false;
  }

  return true;
}

exprt from_integer(
  const mp_integer &int_value,
  const typet &type)
{
  exprt expr;

  expr.clear();
  expr.type()=type;
  expr.id("constant");

  const irep_idt &type_id=type.id();

  if(type_id=="unsignedbv" || type_id=="signedbv")
  {
    expr.value(integer2binary(int_value, bv_width(type)));
    return expr;
  }
  else if(type_id=="bool")
  {
    if(int_value==0)
    {
      expr.make_false();
      return expr;
    }
    else if(int_value==1)
    {
      expr.make_true();
      return expr;
    }
  }

  expr.make_nil();
  return expr;
}

mp_integer power(const mp_integer &base,
                 const mp_integer &exponent)
{
  assert(exponent>=0);

  if(exponent==0)
    return 1;

  mp_integer result(base);
  mp_integer count(exponent-1);

  while(count!=0)
  {
    result*=base;
    --count;
  }

  return result;
}
