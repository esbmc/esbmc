/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/arith_tools.h>
#include <util/fixedbv.h>
#include <util/format_constant.h>
#include <util/ieee_float.h>

std::string format_constantt::operator()(const exprt &expr)
{
  if(expr.is_constant())
  {
    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      mp_integer i;
      if(to_integer(expr, i)) return "(number conversion failed)";

      return integer2string(i);
    }
    else if(expr.type().id()=="fixedbv")
    {
      return fixedbvt(to_constant_expr(expr)).format(*this);
    }
    else if(expr.type().id()=="floatbv")
    {
      return ieee_floatt(to_constant_expr(expr)).format(*this);
    }
  }
  else if(expr.id()=="string-constant")
    return expr.value().as_string();

  return "(format-constant failed: "+expr.id_string()+")";
}

