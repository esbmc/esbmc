// The elements past an array's braced list are initialised as if from an
// empty list: by the default constructor, or by default member initializers.
#include <cassert>
struct C
{
  int v;
  C() : v(9) {}
  C(int x) : v(x) {}
};
struct S
{
  int a = 5;
  int b;
};
int main()
{
  C c[3]{C(1), 2};
  assert(c[1].v == 2 && c[2].v == 9);
  S s[2]{S{1, 2}};
  assert(s[1].a == 5 && s[1].b == 0);
  return 0;
}
