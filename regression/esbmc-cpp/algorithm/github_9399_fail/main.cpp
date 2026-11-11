// #9399: set_symmetric_difference copies the rest of range 1 once range 2 ends
#include <algorithm>
#include <cassert>

int main()
{
  int a[3] = {1, 2, 3};
  int b[1] = {1};
  int o[3];
  int *r = std::set_symmetric_difference(a, a + 3, b, b + 1, o);
  assert(r != o + 2 || o[1] != 3);
  return 0;
}
