#include <assert.h>
int nondet_int(void);

struct S { int *p; int tag; };

int main()
{
  int a = 1, b = 1, c = 2;
  int *q = nondet_int() ? &a : &b;
  struct S t;
  t.p = q;
  t.tag = 0;
  struct S r = t;
  t.p = &c;
  struct S r2 = r;
  assert(*r2.p == 1);
  return 0;
}
