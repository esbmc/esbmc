// #9364: the overloads taking non-pointer iterators behave the same.
#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

int main()
{
  std::vector<int> v{5, 4, 6};
  std::vector<int> w(2);
  std::vector<int>::iterator q =
    std::partial_sort_copy(v.begin(), v.end(), w.begin(), w.end());
  assert(q == w.end() && w[0] == 4 && w[1] == 5);
  q = std::partial_sort_copy(
    v.begin(), v.end(), w.begin(), w.end(), std::greater<int>());
  assert(q == w.end() && w[0] == 6 && w[1] == 5);
  return 0;
}
