/* #10087: a store starting in a whole-byte member and ending in bit-fields. */
#include <assert.h>
struct S { unsigned short h; unsigned short a : 4, b : 12; unsigned int c; };
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  *(unsigned int *)&s[i] = 0x12340005u;
  assert(s[i].b == 0);
  return 0;
}
