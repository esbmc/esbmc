/* assume_nonnull_valid_old_pass:
 * Tests __ESBMC_old() combined with --assume-nonnull-valid.
 *
 * p->x starts as nondet (malloc'd).  The wrapper:
 *   1. malloc's p
 *   2. snapshots p->x into old_snap before the call
 *   3. calls body: p->x += 5
 *   4. ASSERT p->x == old_snap + 5   <- ensures(p->x == __ESBMC_old(p->x) + 5)
 *
 * This must hold for ANY initial nondet value of p->x, proving that
 * old_snapshot correctly captures the pre-state of malloc'd memory.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->x == __ESBMC_old(p->x) + 5);

  p->x = p->x + 5;
}

int main()
{
  S s;
  f(&s);
  return 0;
}
