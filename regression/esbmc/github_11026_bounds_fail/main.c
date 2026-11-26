#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned char src[128] = {0}, dst[128];
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 64 && n <= 65);
  memcpy(dst + off, src, n);
  return 0;
}
