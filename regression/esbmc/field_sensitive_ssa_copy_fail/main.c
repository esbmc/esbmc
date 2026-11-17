#include <assert.h>

int nondet_int(void);

struct S
{
  int a;
  char c;
  long b;
  int *p;
};

struct S mk(int v, int *q)
{
  struct S r;
  r.a = v;
  r.c = 1;
  r.b = 2;
  r.p = q;
  return r;
}

int sum(struct S s)
{
  s.a++;
  return s.a + (int)s.b;
}

struct S rec(int n)
{
  struct S q;
  if (n == 0)
  {
    q.a = 0;
    q.p = 0;
    return q;
  }
  q = rec(n - 1);
  q.a = q.a + 1;
  return q;
}

int main(void)
{
  int x = 0;
  struct S s = mk(nondet_int(), &x);
  __ESBMC_assume(s.a > 0 && s.a < 100);
  struct S t;
  t = s;
  t.a = t.a + 1;
  *t.p = 7;
  int r = sum(t);
  assert(x == 7);
  assert(s.a + 1 == t.a);
  assert(r == s.a + 3);
  struct S w = rec(3);
  assert(w.a == 3);
  return 0;
}
