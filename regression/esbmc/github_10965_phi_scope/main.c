// github #10965
#include <assert.h>
struct in { int *p; };
struct out { struct in a; };
_Bool nondet_bool(void);
int main(void)
{
  int x = 5, y = 5, z = 5;
  int *q = nondet_bool() ? &x : &y;
  struct out tmp, r;
  {
    struct out t;
    t.a.p = q;
    tmp = t;
  }
  r.a.p = &z;
  if (nondet_bool())
    r = tmp;
  assert(*r.a.p == 5); /* holds */
  return 0;
}
