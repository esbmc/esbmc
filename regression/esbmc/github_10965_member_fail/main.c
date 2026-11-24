// github #10965
#include <assert.h>
struct in { int *p; };
struct out { struct in a; };
_Bool nondet_bool(void);
int main(void)
{
  int x = 7, y = 7, z = 5;
  int *q = nondet_bool() ? &x : &y;
  struct out t, tmp;
  t.a.p = q;
  tmp = t;
  t.a.p = &z;
  int *p = t.a.p != 0 ? tmp.a.p : &z;
  assert(*p == 5); /* fails */
  return 0;
}
