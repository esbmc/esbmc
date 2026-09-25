#include <assert.h>

// #7992: with the store to b1 dropped, b1 still read 1 here.
_Bool main_b2;
int main()
{
  _Bool b1 = 1;
  b1 ^= 1;
  assert(b1 == 1);
  b1 = 1;
  __ESBMC_assume(b1 < 4);
  b1 ^= main_b2;
}
