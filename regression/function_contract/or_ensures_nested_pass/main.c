/* or_ensures_nested_pass:
 * Nested ||: ensures(flag==0 || flag==1 || *p == __ESBMC_old(*p) + 5)
 * old() is at depth 2 inside ||. Short-circuit fires for flag==0 and flag==1.
 * Only when flag==2 does the old()-based check actually execute.
 *
 * Tests that collect_old_snapshots correctly finds old_snapshot nested two
 * levels deep inside || control flow, and that the wrapper materializes it.
 */
#include <stddef.h>

void f(int *p, int flag)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(flag == 0 || flag == 1 || flag == 2);
  __ESBMC_ensures(flag == 0 || flag == 1 || *p == __ESBMC_old(*p) + 5);

  if (flag == 2)
    *p = *p + 5;
}

int main() { return 0; }
