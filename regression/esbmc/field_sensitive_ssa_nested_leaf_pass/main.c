#include <assert.h>
struct in { int x; int y; };
struct mid { struct in a; int z; struct in b; };
struct out { struct mid m; char c; };
int nondet_int(void);
int main(void)
{
  struct out o;
  o.m.a.x = 1; o.m.a.y = 2; o.m.z = 3;
  o.m.b = o.m.a;
  o.c = 'a';
  struct in t = o.m.b;
  o.m.b.y = nondet_int();
  struct out p = o;
  p.m.a.x = 7;
  assert(t.x == 1 && t.y == 2);
  assert(p.m.b.y == o.m.b.y);
  assert(o.m.a.x == 1);
  assert(p.m.a.x == 7);
  assert(p.m.z == 3);
  return 0;
}
