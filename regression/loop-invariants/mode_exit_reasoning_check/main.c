/* The assertion needs the invariant AND the negated loop condition:
 * sum == i * 10 together with !(i < 1000) gives i == 1000, hence sum == 10000.
 * Only --loop-invariant-check performs that exit reasoning, so it discharges
 * this regardless of the bound; --loop-invariant falls back to unrolling. */
#include <assert.h>

int main(void)
{
  unsigned int i = 0;
  unsigned int sum = 0;

  __ESBMC_loop_invariant(i <= 1000 && sum == i * 10);
  while (i < 1000)
  {
    sum += 10;
    i++;
  }

  assert(sum == 10000);
  return 0;
}
