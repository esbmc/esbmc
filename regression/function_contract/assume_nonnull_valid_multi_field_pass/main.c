/* assume_nonnull_valid_multi_field_pass:
 * Struct with three fields; requires pins two of them, body sets the third.
 * Verifies that:
 *   (a) requires correctly constrains multiple nondet fields
 *   (b) ensures can reference both constrained and body-set fields
 *
 * Specifically: requires(p->a == 3 && p->b == 4)
 *               body:    p->c = p->a * p->a + p->b * p->b  (3^2 + 4^2 = 25)
 *               ensures: p->c == 25
 */
#include <stddef.h>

typedef struct { int a; int b; int c; } T;

void f(T *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(p->a == 3);
  __ESBMC_requires(p->b == 4);
  __ESBMC_ensures(p->c == 25);

  p->c = p->a * p->a + p->b * p->b;
}

int main()
{
  T t;
  t.a = 3;
  t.b = 4;
  f(&t);
  return 0;
}
