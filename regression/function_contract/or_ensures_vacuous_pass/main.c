/* or_ensures_vacuous_pass:
 * ensures(p != NULL || *p == __ESBMC_old(*p) + 1)
 * With --assume-nonnull-valid, p is always non-null (from malloc + ASSUME).
 * Therefore left side is ALWAYS TRUE -> right side NEVER checked.
 * Body deliberately sets *p = 9999 (totally wrong), but ensures is vacuous.
 *
 * Expected: VERIFICATION SUCCESSFUL (vacuity by design of ||).
 * This documents that a poorly-written ensures with || can be trivially satisfied
 * if the left side is implied by the requires/context.
 */
#include <stddef.h>

void f(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p != NULL || *p == __ESBMC_old(*p) + 1);

  *p = 9999; /* wrong, but never checked due to vacuous left side */
}

int main() { return 0; }
