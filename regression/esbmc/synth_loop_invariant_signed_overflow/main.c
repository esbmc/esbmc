/* Signed counter under --overflow-check. goto_check instruments every
 * instruction guard, including the ones this pass emits, so the synthesised
 * (i - i0) * e would draw overflow claims on arithmetic the user never wrote.
 * Must decline while overflow checking is on. */
#include <assert.h>
int nondet_int();
int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n >= 1 && n <= 10);
  int i = 1, sn = 0;
  while (i <= n) { sn = sn + 2; i++; }
  assert(sn == 2 * n || sn == 0);
  return 0;
}
