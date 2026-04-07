#include <stddef.h>

/* ptr_int_increment_fail:
 * Body adds 2, but ensures claims +1 delta.
 * Must be caught as VERIFICATION FAILED for any concrete initial value.
 */

void increment(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(*p == __ESBMC_old(*p) + 1); /* wrong: body adds 2 */
  *p = *p + 2;
}

int main()
{
  int x = 41;
  increment(&x);
  return 0;
}
