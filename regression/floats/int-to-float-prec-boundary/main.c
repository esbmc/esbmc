/* Round-trip int -> float -> int across the float precision boundary (2^24).
 * Integers above 2^24 are not all representable as float; specifically 2^24+1
 * rounds to 2^24 under IEEE round-to-nearest-even.  After the round-trip the
 * result can therefore never equal 2^24+1. */
#include <assert.h>

int nondet_int(void);

int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 16777216 && i <= 16777218);
  float f = (float)i;
  int j = (int)f;
  assert(j != 16777217);
  return 0;
}
