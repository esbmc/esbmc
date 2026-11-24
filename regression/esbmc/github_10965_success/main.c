// github #10965
#include <assert.h>
struct in { int *p; };
struct out { struct in a; };
_Bool nondet_bool(void);

int main(void)
{
  int x = 7, y = 7, z = 7;
  int *q = nondet_bool() ? &x : &y;
  struct out t, tmp;
  t.a.p = q;
  tmp = t;
  t.a.p = &z;
  struct out r = tmp;
  assert(*r.a.p == 7);
  return 0;
}
