/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "boolbv_type.h"

/*******************************************************************\

Function: get_bvtype

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bvtypet get_bvtype(const typet &type)
{
  if(type.id()=="signedbv")
    return IS_SIGNED;
  else if(type.id()=="unsignedbv")
    return IS_UNSIGNED;
  else if(type.id()=="c_enum" ||
          type.id()=="incomplete_c_enum")
    return IS_C_ENUM;
  else if(type.is_floatbv())
    return IS_FLOAT;
  else if(type.is_fixedbv())
    return IS_FIXED;
  else if(type.id()=="bv")
    return IS_BV;
  else if(type.id()=="verilogbv")
    return IS_VERILOGBV;
  else if(type.id()=="range")
    return IS_RANGE;

  return IS_UNKNOWN;
}
