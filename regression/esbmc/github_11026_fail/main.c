#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned char src[128] = {7}, dst[128];
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 64 && n >= 1 && n <= 64);
  memcpy(dst + off, src, n);
  assert(dst[off] == 7);
  if (n > 1)
    assert(dst[off + 1] == 7);
  return 0;
}
