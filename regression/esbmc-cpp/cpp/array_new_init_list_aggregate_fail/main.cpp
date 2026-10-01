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
  assert(p[1].b != 4);
  delete[] p;
  return 0;
}
