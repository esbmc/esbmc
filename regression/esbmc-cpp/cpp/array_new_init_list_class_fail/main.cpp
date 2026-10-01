// Each element of an array new-expression's braced list runs its own
// constructor: the second element is not built by the first one's.
#include <cassert>
struct C
{
  int v;
  C(int x) : v(x) {}
};
int main()
{
  C *p = new C[2]{C(1), C(2)};
  assert(p[1].v == 1);
  delete[] p;
  return 0;
}
