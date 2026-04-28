/* or_requires_content_pass:
 * requires(p == NULL || *p > 0): if p is non-null, then *p must be positive.
 * With --assume-nonnull-valid, p is always non-null, so requires reduces to:
 *   ASSUME(false || *p > 0) = ASSUME(*p > 0)
 * This constrains the nondet malloc'd value.
 * ensures(*p > 0): holds because requires already guaranteed it, body is empty.
 *
 * Tests || in requires with pointer dereference on the right side,
 * where the left side (NULL check) is always false in harness mode.
 */
#include <stddef.h>

void f(int *p)
{
  __ESBMC_requires(p == NULL || *p > 0);
  __ESBMC_ensures(*p > 0);
  /* empty body — requires constraint must carry through to ensures */
}

int main() { return 0; }
