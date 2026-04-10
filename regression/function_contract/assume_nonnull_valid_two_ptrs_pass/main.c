/* assume_nonnull_valid_two_ptrs_pass:
 * Both pointer parameters get separate malloc'd objects.
 * Each is independently set and verified via ensures.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_ensures(p->x == 10);
  __ESBMC_ensures(q->x == 20);

  p->x = 10;
  q->x = 20;
}

int main()
{
  S a, b;
  f(&a, &b);
  return 0;
}
