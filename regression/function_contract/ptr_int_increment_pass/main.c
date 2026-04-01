/* ptr_int_increment_pass:
 * Verifies function contracts work with int* (primitive pointer, not struct).
 * Uses __ESBMC_old(*p) to express the post-state in terms of pre-state.
 * Caller provides a concrete stack variable — no --assume-nonnull-valid needed.
 */
#include <assert.h>
#include <stddef.h>

void increment(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(*p == __ESBMC_old(*p) + 1);
  (*p)++;
}

int main()
{
  int x = 41;
  increment(&x);
  assert(x == 42);
  return 0;
}
