#include <assert.h>

struct S
{
  int a;
  int b;
};

int main(void)
{
  int v = nondet_int() | 1;
  struct S s = {v, nondet_int()};
  assert(s.b != 5);
  return 0;
}
