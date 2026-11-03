// Elements past a braced list are value-initialised ([dcl.init.aggr]/5).
#include <cassert>
int main()
{
  int *p = new int[4]{7, 8};
  assert(p[3] != 0);
  delete[] p;
  return 0;
}
