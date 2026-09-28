/* Signed counter with a SYMBOLIC addend. That needs the two-disjunct bound,
 * which is only establishable for an unsigned counter starting at 0 or 1, so
 * the relaxation must not apply here. Declines. */
#include <assert.h>
int nondet_int();
int main(void)
{
  int n = nondet_int(), a = nondet_int();
  __ESBMC_assume(n >= 1 && n <= 6);
  __ESBMC_assume(a >= 0 && a <= 3);
  int i = 1, sn = 0;
  while (i <= n) { sn = sn + a; i++; }
  assert(sn == n * a || sn == 0);
  return 0;
}
