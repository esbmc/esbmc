#include <assert.h>
int nondet_int(void);

struct P { int *p; };
struct S { struct P f; int tag; };

int main()
{
  int a = 1, b = 1;
  struct P s;
  s.p = nondet_int() ? &a : &b;
  struct S r;
  {
    struct S t;
    t.f = s;
    t.tag = 0;
    r = t;
  }
  struct S r2 = r;
  assert(*r2.f.p == 1);
  return 0;
}
