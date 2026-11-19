#include <assert.h>

int nondet_int(void);

struct S
{
  int a;
  int b;
};

struct S make(int x)
{
  struct S r;
  r.a = 1;
  r.b = x + 1;
  return r;
}

int main(void)
{
  int x = nondet_int();
  struct S s;
  struct S *p = &s;
  *p = make(x);
  assert(s.a == 1);
  assert(s.b == x + 1);
  return 0;
}
