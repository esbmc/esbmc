/* ptr_cross_req_pass:
 * Two separate malloc'd struct pointers p and q.
 * requires(p->x == q->x) constrains BOTH nondet malloc'd values to be equal.
 * Empty body leaves them unchanged, so ensures(p->x == q->x) must hold.
 * Verifies that requires can constrain values across two independent mallocs.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_requires(p->x == q->x);
  __ESBMC_ensures(p->x == q->x);
  /* empty body: values unchanged, equality must persist */
}

int main()
{
  S a = {5}, b = {5};
  f(&a, &b);
  return 0;
}
