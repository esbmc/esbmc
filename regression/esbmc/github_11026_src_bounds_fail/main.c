#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned char src[16] = {7}, dst[16];
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 16 && n <= 8);
  memcpy(dst + 3, src + off, n);
  return 0;
}
