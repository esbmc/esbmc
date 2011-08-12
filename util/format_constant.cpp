/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "format_constant.h"
#include "arith_tools.h"
#include "fixedbv.h"
#include "ieee_float.h"

/*******************************************************************\

Function: format_constantt::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string format_constantt::operator()(const exprt &expr)
{
  if(expr.is_constant())
  {
    if(expr.type().id()=="natural" ||
       expr.type().id()=="integer" ||
       expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      mp_integer i;
      if(to_integer(expr, i)) return "(number conversion failed)";

      return integer2string(i);
    }
    else if(expr.type().is_fixedbv())
    {
      return fixedbvt(expr).format(*this);
    }
    else if(expr.type().is_floatbv())
    {
      return ieee_floatt(expr).format(*this);
    }
  }
  else if(expr.id()=="string-constant")
    return expr.value().as_string();

  return "(format-constant failed: "+expr.id_string()+")";
}

