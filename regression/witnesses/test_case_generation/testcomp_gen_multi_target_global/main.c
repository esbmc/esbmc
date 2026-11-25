#include <assert.h>

extern int __VERIFIER_nondet_int(void);

struct pair
{
  int a, b;
};

struct pair g1, g2;

int main()
{
  struct pair *p = __VERIFIER_nondet_int() ? &g1 : &g2;
  p->b = __VERIFIER_nondet_int();
  if (p == &g1 && g1.b == 3)
    assert(0);
  return 0;
}
