/* assume_nonnull_valid_two_ptrs_fail:
 * Body sets p->x=10 and q->x=20, but ensures claims p->x == q->x.
 * Since 10 != 20 this must be caught as VERIFICATION FAILED.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_ensures(p->x == q->x); /* 10 != 20 */

  p->x = 10;
  q->x = 20;
}

int main()
{
  S a, b;
  f(&a, &b);
  return 0;
}
