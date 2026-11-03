// Each element of an array new-expression's braced list runs its own
// constructor, once.
#include <cassert>
int live = 0;
struct C
{
  int v;
  C(int x) : v(x) { ++live; }
  ~C() { --live; }
};
int main()
{
  C *p = new C[2]{C(1), C(2)};
  assert(live == 2 && p[0].v == 1 && p[1].v == 2);
  delete[] p;
  assert(live == 0);
  return 0;
}
