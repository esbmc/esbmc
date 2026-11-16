// #9364: partial_sort_copy of three inputs into three outputs returns o + 3.
#include <algorithm>
#include <cassert>

int main()
{
  int a[3] = {3, 1, 2};
  int o[3];
  int *r = std::partial_sort_copy(a, a + 3, o, o + 3);
  assert(r != o + 3);
  return 0;
}
