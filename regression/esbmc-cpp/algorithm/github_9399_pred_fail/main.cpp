// #9399: set_symmetric_difference copies the rest of range 1 once range 2 ends
#include <algorithm>
#include <cassert>
#include <functional>

int main()
{
  int c[3] = {3, 2, 1};
  int d[1] = {3};
  int o[3];
  int *q =
    std::set_symmetric_difference(c, c + 3, d, d + 1, o, std::greater<int>());
  assert(q != o + 2 || o[1] != 1);
  return 0;
}
