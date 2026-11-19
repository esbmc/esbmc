#include <assert.h>

int nondet_int(void);

struct S
{
  int n;
  int x;
};

int main(void)
{
  struct S s;
  s.n = 3;
  s.x = nondet_int();
  struct S t = s;
  int *p = &t.x;
  int i = 0;
  while (i < t.n)
    i++;
  assert(i == 3);
  assert(*p == s.x);
  return 0;
}
