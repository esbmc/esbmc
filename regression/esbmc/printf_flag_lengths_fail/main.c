#include <stdio.h>

unsigned nondet_uint(void);

int main(void)
{
  unsigned x = nondet_uint();
  int n = printf("%#x", x);
  __ESBMC_assert(n <= 8, "a %#x of an unsigned int prints at most 8 chars");
  return 0;
}
