// A braced list of aggregates: the listed elements are stored and the rest
// value-initialised.
#include <cassert>
struct S
{
  int a, b;
};
int main()
{
  S *p = new S[3]{{1, 2}, {3, 4}};
  assert(p[0].b == 2 && p[1].a == 3);
  assert(p[2].a == 0 && p[2].b == 0);
  delete[] p;
  return 0;
}
