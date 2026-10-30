// #9405
#include <algorithm>
#include <cassert>

int main()
{
  int a[3] = {2, 0, 1};
  std::partial_sort(a, a + 2, a + 3);
  assert(!(a[0] == 0 && a[1] == 1));
  return 0;
}
