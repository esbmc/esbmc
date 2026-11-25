#include <assert.h>
int nondet_int(void);
struct S { int a; int b; };
void f(struct S p, int x)
{
  assert(p.b == x);
}
int main(void)
{
  int x = nondet_int();
  struct S s;
  s.a = 1;
  s.b = x + 1;
  struct S t = s;
  struct S *q = &t;
  f(t, x);
  return q->a - 1;
}
