/* Regression: GitHub #7585 -- the downgrade added in #7491 was decided by the
 * claim's position rather than by whether the invariant pins the
 * counterexample. Here `sn == (i - 1) * a` together with the exit condition
 * gives `i == n + 1` and so `sn == n * a`, which the assertion below states, so a strong invariant proves it where
 * every n and a: no abstract state satisfies it, so the violation is real. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;

  __ESBMC_assume(n >= 1);

  __ESBMC_loop_invariant(
    (i <= (uint64_t)n || i == (uint64_t)n + 1) && i >= 1 &&
    sn == (i - 1) * a);

  while (i <= n)
  {
    sn = sn + a;
    i++;
  }

  assert(sn == (uint64_t)n * a);
  return 0;
}
