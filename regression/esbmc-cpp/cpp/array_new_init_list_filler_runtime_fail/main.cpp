// With an element count that is not a constant expression, the elements past
// the braced list take the filler, not zero.
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
  assert(p[2].a == 0);
  delete[] p;
  return 0;
}
