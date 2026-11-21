/* #10905: pointers with the same address compare equal, also when one is made
 * from an integer and then offset. */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
int main(void)
{
  uint32_t buf[1];
  unsigned off = nondet_uint();
  __ESBMC_assume(off < 4);
  uint32_t *p1 = buf;
  uint32_t *p2 = (uint32_t *)(uintptr_t)4 + off;
  if ((intptr_t)p1 == (intptr_t)p2)
    assert(p1 == p2);
  if (p1 == p2)
    assert(p2 - p1 == 0);
  uint32_t *p3 = (uint32_t *)(uintptr_t)4 + 3;
  if ((intptr_t)p1 == (intptr_t)p3)
    assert(p1 == p3);
  return 0;
}
