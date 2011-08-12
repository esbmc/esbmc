/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>

#include "bitvector.h"

/*******************************************************************\

Function: bv_sem

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bv_semt bv_sem(const typet &type)
{
  if(type.id()=="bv")
    return BV_NONE;
  else if(type.id()=="unsignedbv")
    return BV_UNSIGNED;
  else if(type.is_signedbv())
    return BV_SIGNED;

  return BV_UNKNOWN;
}

/*******************************************************************\

Function: bv_width

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}


