/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/bitvector.h>
#include <cstdlib>

unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}


