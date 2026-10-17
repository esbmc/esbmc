/* github #8658 */
#include <limits.h>
int nondet_int();
int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a == INT_MIN);
  int r = a + a;
  return r;
}
