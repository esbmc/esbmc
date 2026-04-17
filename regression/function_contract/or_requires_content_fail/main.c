/* or_requires_content_fail:
 * Same requires(p == NULL || *p > 0), but ensures(*p > 1) — stronger than requires.
 * requires only guarantees *p > 0, which includes *p == 1.
 * For *p == 1: requires satisfied, but ensures *p > 1 fails (1 is not > 1).
 * Expected: VERIFICATION FAILED.
 *
 * Confirms that || in requires correctly constrains content but cannot
 * over-satisfy a stronger ensures.
 */
#include <stddef.h>

void f(int *p)
{
  __ESBMC_requires(p == NULL || *p > 0);
  __ESBMC_ensures(*p > 1); /* stronger than requires guarantees */
  /* empty body */
}

int main() { return 0; }
