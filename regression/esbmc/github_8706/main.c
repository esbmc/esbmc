/* github #8706: Bitwuzla flagged unsigned / and % as signed division
 * overflow when the operands' bit patterns are INT_MIN and -1. */
#include <assert.h>
unsigned nondet_uint();
int main(void)
{
  unsigned a = nondet_uint(), b = nondet_uint();
  __ESBMC_assume(a == 0x80000000u && b == 0xffffffffu);
  unsigned q = a / b;
  unsigned r = a % b;
  assert(q == 0 && r == 0x80000000u);
  return 0;
}
