/* or_ensures_old_conditional_pass:
 * __ESBMC_old() is inside the RIGHT side of ||.
 * ensures(delta == 0 || *p == __ESBMC_old(*p) + delta):
 *   - delta==0: left TRUE, short-circuit fires, old() NOT evaluated in original
 *   - delta>0:  left FALSE, old() IS evaluated, body must satisfy *p = old+delta
 *
 * CRITICAL TEST: collect_old_snapshots_from_body must find the old_snapshot
 * even though it's inside a conditional IF branch (due to || short-circuit).
 * The wrapper materializes it UNCONDITIONALLY at call entry.
 * All nondet delta >= 0 paths must pass.
 */
#include <stddef.h>

void f(int *p, int delta)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(delta >= 0);
  __ESBMC_ensures(delta == 0 || *p == __ESBMC_old(*p) + delta);

  *p = *p + delta;
}

int main() { return 0; }
