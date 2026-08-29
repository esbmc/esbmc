// The struct-member form, which is what an elementwise polynomial update looks
// like. `__ESBMC_assigns(p->coeffs)` decays the same way a bare array does.
#include <assert.h>

typedef struct
{
  int coeffs[4];
} poly;

void f(poly *p)
{
  __ESBMC_assigns(p->coeffs);
  __ESBMC_ensures(
    p->coeffs[0] == 1 && p->coeffs[1] == 1 && p->coeffs[2] == 1 &&
    p->coeffs[3] == 1);
  for (int i = 0; i < 4; i++)
    p->coeffs[i] = 1;
}

int main(void)
{
  poly q;
  for (int i = 0; i < 4; i++)
    q.coeffs[i] = 0;
  f(&q);
  assert(q.coeffs[3] == 1); /* holds under the contract */
  assert(0);                /* reachable */
  return 0;
}
