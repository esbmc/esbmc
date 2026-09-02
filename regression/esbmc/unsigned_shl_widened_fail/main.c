#include <assert.h>

typedef unsigned int u32;

unsigned int nondet_uint();

/* No widening here, so the shift genuinely discards set bits and the
   unsigned overflow check must still fire. */
int main()
{
  u32 hi = nondet_uint();
  u32 v = hi << 8;
  assert(v >= 0);
  return 0;
}
