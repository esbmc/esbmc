#include <cassert>

int main()
{
  // [expr.new]/24: an array element type is value-initialised too.
  int(*a)[3] = new int[2][3]();
  assert(a[1][2] != 0);
  delete[] a;
  return 0;
}
