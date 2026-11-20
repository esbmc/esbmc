#include <assert.h>

struct S
{
  int a;
  int b;
  int *q;
};

void set(struct S *p, int v)
{
  p->a = v;
  p->b = v + 1;
}

int main(void)
{
  struct S s;
  int x = 5;
  s.q = &x;
  set(&s, 3);
  struct S *r = &s;
  assert(s.a == 3 && r->b == 4 && *r->q == 5);
  *r = (struct S){1, 2, &x};
  struct S t = *r;
  assert(s.b == 2 && t.a == 1 && *t.q == 5);
  return 0;
}
