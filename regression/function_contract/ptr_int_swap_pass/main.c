/* ptr_int_swap_pass:
 * Classic integer swap via two int* parameters.
 * The ensures is a cross-reference: after swap,
 *   *a == pre-state(*b)  and  *b == pre-state(*a).
 *
 * Uses --assume-nonnull-valid so that a and b receive separate
 * malloc'd ints with nondet initial values (non-aliasing guaranteed).
 * The ensures must hold for ALL nondet initial combinations.
 */
#include <stddef.h>

void swap(int *a, int *b)
{
  __ESBMC_requires(a != NULL && b != NULL);
  __ESBMC_ensures(*a == __ESBMC_old(*b) && *b == __ESBMC_old(*a));

  int tmp = *a;
  *a = *b;
  *b = tmp;
}

int main()
{
  int x = 1, y = 2;
  swap(&x, &y);
  return 0;
}
