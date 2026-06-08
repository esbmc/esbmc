/* Soundness check: the modelled return value of printf with a nondet
   argument must be provably within the actual possible range.
   printf("%x", v) for a 32-bit unsigned prints 1–8 hex chars, so
   assert(r >= 1 && r <= 8) must hold. */
#include <assert.h>
#include <stdio.h>
extern unsigned nondet_uint(void);
int main()
{
#ifndef _WIN32
  unsigned v = nondet_uint();
  int r = printf("%x", v);
  assert(r >= 1 && r <= 8);
#endif
}
