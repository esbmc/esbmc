/* The accumulator is written `s = a + s`, with the target on the right of the
 * addition. That is the same self-increment as `s = s + a` and must be
 * recognised as one. */
#include <stdint.h>
#include <assert.h>

int main(void)
{
  uint32_t n;
  uint64_t a;
  __ESBMC_assume(n >= 1 && n <= 3);

  uint64_t i = 1;
  uint64_t s = 0;

  while (i <= n)
  {
    s = a + s;
    i++;
  }

  assert(s == (uint64_t)n * a);
  return 0;
}
