/* The unrolled model covers only the bytes the objects actually have, so a
   length that could exceed them must still be reported rather than dropped. */
#include <string.h>

int main()
{
  char src[8];
  char dst[4];

  unsigned long n = nondet_ulong();
  __ESBMC_assume(n <= 8);

  memcpy(dst, src, n); /* n may exceed dst's 4 bytes */
  return 0;
}
