/* ptr_int_swap_fail:
 * Broken swap: only copies b->a, never restores old *a into *b.
 * After body: *a == old(*b) (correct) but *b == old(*b) (wrong).
 * Must be caught as VERIFICATION FAILED.
 */
#include <stddef.h>

void swap(int *a, int *b)
{
  __ESBMC_requires(a != NULL && b != NULL);
  __ESBMC_ensures(*a == __ESBMC_old(*b) && *b == __ESBMC_old(*a));

  *a = *b; /* copies b to a, but loses original *a */
}

int main()
{
  int x = 1, y = 2;
  swap(&x, &y);
  return 0;
}
