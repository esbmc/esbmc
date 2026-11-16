// #9405: comparator overload.
#include <algorithm>
#include <cassert>
#include <functional>

int main()
{
  int a[4] = {1, 3, 0, 2};
  std::partial_sort(a, a + 2, a + 4, std::greater<int>());
  assert(a[0] == 3 && a[1] == 2);
  return 0;
}
