// github #10965
#include <assert.h>
struct in { int *p; };
struct out { struct in a; };
_Bool nondet_bool(void);
int main(void)
{
  int x = 7, y = 7, z = 5;
  int *q = nondet_bool() ? &x : &y;
  struct out t, tmp, u;
  t.a.p = q;
  tmp = t;
  t.a.p = &z;
  u.a.p = &z;
  struct out *pp = nondet_bool() ? &t : &u;
  *pp = tmp;
  assert(*t.a.p == 5); /* fails when pp == &t */
  return 0;
}
