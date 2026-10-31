// #9370: the comparator overloads of set_difference and set_union must
// order elements by the comparator, not by operator<.
#include <algorithm>
#include <cassert>
#include <functional>

int main()
{
  int a[3] = {3, 2, 1};
  int b[1] = {2};
  int o[4];
  int *r = std::set_difference(a, a + 3, b, b + 1, o, std::greater<int>());
  assert(r == o + 2 && o[0] == 3 && o[1] == 1);

  int c[2] = {4, 2};
  r = std::set_union(a, a + 3, c, c + 2, o, std::greater<int>());
  assert(r == o + 4 && o[0] == 4 && o[1] == 3 && o[2] == 2 && o[3] == 1);
  return 0;
}
