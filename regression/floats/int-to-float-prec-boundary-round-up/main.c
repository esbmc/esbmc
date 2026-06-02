/* Round-trip int -> float -> int across the float precision boundary (2^24),
 * round-up tie case.
 * At binade k=1 (values in [2^24, 2^25)), representable floats are spaced by 2.
 * 16777219 = 2^24 + 3 is halfway between 16777218 and 16777220.
 * Its floor significand index is 16777218/2 = 8388609 (odd), so RNE rounds UP
 * to 16777220.  After the round-trip the result cannot equal 16777219. */
#include <assert.h>

int nondet_int(void);

int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 16777218 && i <= 16777220);
  float f = (float)i;
  int j = (int)f;
  assert(j != 16777219);
  return 0;
}
