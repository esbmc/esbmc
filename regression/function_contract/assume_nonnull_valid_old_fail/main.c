/* assume_nonnull_valid_old_fail:
 * Body adds 3 to p->x, but ensures claims +5.
 * The mismatch must be detected as VERIFICATION FAILED for any
 * initial nondet value of p->x (from malloc).
 *
 * This ensures that old_snapshot is NOT trivially satisfiable when
 * the ensures delta is wrong.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->x == __ESBMC_old(p->x) + 5); /* wrong: body adds 3 */

  p->x = p->x + 3;
}

int main()
{
  S s;
  f(&s);
  return 0;
}
