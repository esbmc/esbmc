#include <assert.h>
typedef unsigned int u32;
unsigned int nondet_uint(void);
int main(void)
{
  u32 a = nondet_uint();
  __ESBMC_assume(a >= 0x80000000u);
  /* logical shift: a>>1 <= 0x7FFFFFFF.  arithmetic shift would give >=0xC0000000 */
  assert((a >> 1) < 0x80000000u);
  u32 c = a;
  c >>= 1;
  assert(c < 0x80000000u);
  return 0;
}
