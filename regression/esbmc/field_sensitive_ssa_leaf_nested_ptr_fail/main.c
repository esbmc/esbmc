#include <assert.h>
struct in { int x; int y; };
struct out { struct in a; struct in b; };
void f(struct out *p) { p->a.y = 5; p->b = p->a; }
int main(void)
{
  struct out o = {{1,2},{3,4}};
  f(&o);
  assert(o.b.y == 5);
  assert(o.b.x == 1);
  assert(o.a.x == 2); // fails
  return 0;
}
