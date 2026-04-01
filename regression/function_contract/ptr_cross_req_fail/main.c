/* ptr_cross_req_fail:
 * requires(p->x == q->x) constrains the two nondet values to be equal.
 * Body increments p->x by 1, breaking the equality.
 * ensures(p->x == q->x) must be VERIFICATION FAILED.
 * Demonstrates that requires-constrained state can still be violated by body.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_requires(p->x == q->x);
  __ESBMC_ensures(p->x == q->x);

  p->x = p->x + 1; /* breaks equality: p->x = old+1, q->x = old */
}

int main()
{
  S a = {5}, b = {5};
  f(&a, &b);
  return 0;
}
