#include <assert.h>

// #7992: `b1 ^= 1` is lowered to `(signed int)b1 = (signed int)b1 ^ 1`, and
// GCSE replaced that target with a CSE symbol, dropping the store to b1.
_Bool main_b2;
int main()
{
  _Bool b1 = 1;
  b1 ^= 1;
  assert(b1 == 0);
  b1 = 1;
  __ESBMC_assume(b1 < 4);
  b1 ^= main_b2;
}
