#include <assert.h>
int nondet_int(void);

struct S { int *p; int n; };

int main()
{
  int a = 1, b = 2, c = 3;
  int *q = nondet_int() ? &a : &b;
  struct S t;
  t.p = q;
  t.n = 3;
  struct S r = t;
  t.p = &c;
  t.n = 0;
  struct S r2 = r;
  int s = 0;
  for (int i = 0; i < r2.n; ++i)
    ++s;
  assert(s == 3);
  assert(*r2.p == 1 || *r2.p == 2);
  return 0;
}
