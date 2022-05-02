/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstdlib>
#include <util/bitvector.h>

unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}
