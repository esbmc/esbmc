// github #10965
#include <assert.h>
struct in { int *p; };
struct out { struct in a; };
_Bool nondet_bool(void);
int main(void)
{
  int x = 7, y = 7, z = 5;
  int *q = nondet_bool() ? &x : &y;
  struct out t, tmp, r;
  t.a.p = q;
  tmp = t;
  t.a.p = &z;
  r.a.p = &z;
  if (nondet_bool())
    r = tmp;
  assert(*r.a.p == 5); /* fails when branch taken */
  return 0;
}
