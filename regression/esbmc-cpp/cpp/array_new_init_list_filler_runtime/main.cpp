// With an element count that is not a constant expression, the elements past
// the braced list take the filler: here S's default member initializer.
#include <cassert>
struct S
{
  int a = 5;
  int b;
};
int main()
{
  unsigned n = 3;
  S *p = new S[n]{S{1, 2}};
  assert(p[0].a == 1 && p[2].a == 5 && p[2].b == 0);
  delete[] p;
  return 0;
}
