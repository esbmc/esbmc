// memcpy between a pointer object and a one-pointer aggregate: the byte range
// matches on both sides, but only at different depths, so do_memcpy_expression
// has to pick the sub-object pair by type rather than by depth.
#include <assert.h>
#include <string.h>

struct S
{
  int *p;
};

union U
{
  long i;
  int *p;
};

struct W1
{
  union U u;
};

struct W2
{
  union U u;
};

int main()
{
  int x = -2;
  struct S s = {&x};
  int *q;
  memcpy(&q, &s, sizeof q);
  assert(q == &x);
  assert(*q == -2);

  int y = 7;
  int *r = &y;
  struct S t;
  memcpy(&t, &r, sizeof r);
  assert(t.p == &y);
  assert(*t.p == 7);

  int z = 11;
  int *a[1] = {&z};
  struct S u;
  memcpy(&u, &a, sizeof a);
  assert(u.p == &z);
  assert(*u.p == 11);

  int w = 42;
  struct W2 b;
  b.u.p = &w;
  struct W1 c;
  memcpy(&c, &b, sizeof b);
  assert(c.u.p == &w);
  assert(*c.u.p == 42);

  return 0;
}
