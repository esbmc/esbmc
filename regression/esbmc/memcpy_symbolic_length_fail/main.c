/* A memcpy whose length is not a constant is modelled by unrolling to the
   objects' own widths and guarding each byte with i < n, instead of deferring
   to the C byte loop (which unwinds --unwind times per call). */
#include <string.h>
#include <assert.h>

int main()
{
  char src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  char dst[8] = {7, 7, 7, 7, 7, 7, 7, 7};

  unsigned long n = nondet_ulong();
  __ESBMC_assume(n == 3);

  void *r = memcpy(dst, src, n);

  assert(r == (void *)dst);
  assert(dst[0] == 1);
  assert(dst[2] == 3);
  /* bytes at or past n are untouched */
  assert(dst[3] == 4);
  assert(dst[7] == 7);
  return 0;
}
