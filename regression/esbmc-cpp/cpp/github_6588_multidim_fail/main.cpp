#include <cassert>

int main()
{
  // An array element type is left indeterminate rather than zero-filled;
  // this must still produce a verdict rather than an internal error.
  int(*a)[3] = new int[2][3]();
  assert(a[0][0] == 0);
  delete[] a;
  return 0;
}
