#include <string.h>

unsigned nondet_uint(void);

int main()
{
  char d[8];
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 8 && n <= 9 - off);
  memset(d + off, 0, n);
  return 0;
}
