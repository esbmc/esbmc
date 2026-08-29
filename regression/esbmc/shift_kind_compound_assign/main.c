#include <assert.h>
typedef unsigned int u32;
unsigned int nondet_uint(void);
int nondet_int(void);

int main(void)
{
  u32 a = nondet_uint();
  __ESBMC_assume(a >= 0x80000000u);
  a >>= 1;
  /* logical shift: a <= 0x7FFFFFFF.  arithmetic would give >= 0xC0000000 */
  assert(a < 0x80000000u);

  unsigned int b = nondet_uint();
  __ESBMC_assume(b >= 0x80000000u);
  b >>= 1;
  assert(b < 0x80000000u);

  int s = nondet_int();
  __ESBMC_assume(s < 0);
  s >>= 1;
  /* arithmetic shift keeps the sign; a logical one would make it non-negative */
  assert(s < 0);

  return 0;
}
