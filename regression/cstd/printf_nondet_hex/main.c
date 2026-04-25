/* Case B: non-constant %x argument.
   printf("%x", v) for a 32-bit unsigned v prints 1–8 hex digits,
   so the return value r is in [1,8].  assert(r==8) must fail because
   the counterexample v==0 produces "0" (1 char, not 8). */
#include <assert.h>
#include <stdio.h>
extern unsigned nondet_uint(void);
int main()
{
#ifndef _WIN32
  unsigned v = nondet_uint();
  int r = printf("%x", v);
  assert(r == 8);
#else
  assert(0);
#endif
}
