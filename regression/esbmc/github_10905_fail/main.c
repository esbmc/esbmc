/* #10905: a pointer made from an integer still names no object when it
 * compares equal to one, so reading through it is reported. */
#include <stdint.h>
unsigned nondet_uint(void);
int main(void)
{
  uint32_t buf[1] = {7};
  unsigned off = nondet_uint();
  __ESBMC_assume(off < 4);
  uint32_t *p1 = buf;
  uint32_t *p2 = (uint32_t *)(uintptr_t)4 + off;
  if (p1 == p2)
    return *p2;
  return 0;
}
