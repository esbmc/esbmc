#include <assert.h>

/* The negative half of builtin_overflow_generic: the model must report the
 * overflow rather than silently returning a nondet answer that satisfies
 * whatever the caller assumed. */
int main(void)
{
  int r;
  assert(!__builtin_mul_overflow(0x7fffffff, 2, &r));
  return 0;
}
