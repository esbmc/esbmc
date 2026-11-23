#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
int main(void)
{
  uint32_t buf[1];
  unsigned off = nondet_uint();
  __ESBMC_assume(off < 4);
  uint32_t *p1 = buf;
  __ESBMC_assume((uintptr_t)p1 != 4);
  uint32_t *p2 = (uint32_t *)(uintptr_t)4 + off;
  if ((intptr_t)p1 == (intptr_t)p2)
    assert(p1 != p2);
  return 0;
}
