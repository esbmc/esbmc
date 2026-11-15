#include <assert.h>

int nondet_int(void);

struct S
{
  int a;
  int b;
};

int main(void)
{
  struct S t;
  t.a = nondet_int();
  t.b = 2;
  struct S u;
  u = t;
  assert(u.a != 1);
  return 0;
}
