/* #10891: an over-aligned struct read at a symbolic index into a uint32_t
 * array is misaligned for odd i (C17 6.3.2.3p7). */
#include <stdint.h>
unsigned nondet_uint(void);
struct __attribute__((aligned(8))) S { uint32_t a; };
int main(void)
{
  uint32_t buf[4] __attribute__((aligned(8))) = {1, 2, 3, 4};
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 3);
  struct S *p = (struct S *)&buf[i];
  struct S s = *p;
  return s.a;
}
