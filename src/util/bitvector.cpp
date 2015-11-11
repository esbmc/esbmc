/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>

#include "bitvector.h"

unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}


