// The elements past an array new-expression's braced list are initialised by
// the class's default constructor, not zeroed.
#include <cassert>
struct C
{
  int v;
  C() : v(9) {}
  C(int x) : v(x) {}
};
int main()
{
  C *p = new C[3]{C(1)};
  assert(p[2].v == 0);
  delete[] p;
  return 0;
}
