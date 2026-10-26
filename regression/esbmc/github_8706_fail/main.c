/* github #8706 */
#include <assert.h>
#include <limits.h>
unsigned nondet_uint();
int nondet_int();
int main(void)
{
  unsigned a = nondet_uint(), b = nondet_uint();
  __ESBMC_assume(a == 0x80000000u && b == 0xffffffffu);
  unsigned q = a / b;
  assert(q == 1);
  int x = nondet_int(), y = nondet_int();
  __ESBMC_assume(x == INT_MIN && y == -1);
  int s = x / y;
  return s;
}
