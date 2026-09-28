#include <cstdlib>
#include <util/arith/bitvector.h>

unsigned bv_width(const typet &type)
{
  return atoi(type.width().c_str());
}
