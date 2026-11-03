// #9394: "%*d" with 6 and 42 prints "    42", and a width the program does
// not fix, or a huge one, leaves the length open.
#include <assert.h>
#include <stdio.h>

extern int __VERIFIER_nondet_int(void);

int main(void)
{
  char b[16];
  int r = snprintf(b, sizeof b, "%*d", 6, 42);
  assert(r != 6);

  int w = __VERIFIER_nondet_int();
  __ESBMC_assume(w >= 0 && w <= 8);
  r = snprintf(b, sizeof b, "%*d", w, 42);
  assert(r == 2);

  r = snprintf(b, sizeof b, "%*d", -2147483647 - 1, 42);
  assert(r == 2);
  return 0;
}
