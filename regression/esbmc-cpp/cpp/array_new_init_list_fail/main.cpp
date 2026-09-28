// An array new-expression's braced list initialises its elements in order.
#include <cassert>
int main()
{
  int *p = new int[2]{1, 2};
  assert(p[1] != 2);
  delete[] p;
  return 0;
}
