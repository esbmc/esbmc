#include <assert.h>

struct S
{
  int a, b;
};

int nondet_int();

int main()
{
  struct S s;
  s.a = nondet_int();
  s.b = 2;
  assert(s.a != 5);
  return 0;
}
