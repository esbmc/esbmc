/* assume_nonnull_valid_req_content_fail:
 * requires(p->x == 5) and body does p->x += 10, so p->x == 15.
 * ensures claims p->x == 20 which is wrong -> VERIFICATION FAILED.
 *
 * This proves that:
 *   (a) requires really constrains the nondet malloc content (not vacuous)
 *   (b) the ensures violation is detected despite the requires constraint
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(p->x == 5);
  __ESBMC_ensures(p->x == 20); /* wrong: 5 + 10 = 15, not 20 */

  p->x = p->x + 10;
}

int main()
{
  S s;
  s.x = 5;
  f(&s);
  return 0;
}
