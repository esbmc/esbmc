#include <assert.h>
int nondet_int(void);
struct T
{
  int *p;
  int k;
};
struct T g;
struct T mk(int c, int *a, int *b)
{
  struct T t;
  t.k = 0;
  if (c)
    t.p = a;
  else
    t.p = b;
  return t;
}
int main(void)
{
  int a, b;
  g = mk(nondet_int(), &a, &b);
  assert(g.k == 0);
  return *g.p;
}
