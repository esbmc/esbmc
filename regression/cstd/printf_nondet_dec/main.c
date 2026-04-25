/* Case C: non-constant %d argument.
   ESBMC models %d as signed int, so a 32-bit value spans -2147483648..2147483647.
   The return value r is in [1,11] (11 for "-2147483648").  assert(r>20) must fail. */
#include <assert.h>
#include <stdio.h>
extern unsigned nondet_uint(void);
int main()
{
#ifndef _WIN32
  unsigned v = nondet_uint();
  int r = printf("%d", v);
  assert(r > 20);
#else
  assert(0);
#endif
}
