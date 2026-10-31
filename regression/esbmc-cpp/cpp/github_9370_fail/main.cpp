// #9370: the comparator overload of set_difference keeps 3 and 1 and drops 2.
#include <algorithm>
#include <cassert>
#include <functional>

int main()
{
  int a[3] = {3, 2, 1};
  int b[1] = {2};
  int o[3];
  int *r = std::set_difference(a, a + 3, b, b + 1, o, std::greater<int>());
  assert(r != o + 2 || o[0] != 3 || o[1] != 1);
  return 0;
}
