#include <iterator>
#include <vector>
#include <cassert>

int main()
{
  // [iterator.range]
  int a[4] = {1, 2, 3, 4};
  assert(std::size(a) == 4);
  assert(!std::empty(a));
  assert(std::data(a) == &a[0]);
  static_assert(std::size(a) == 4, "size of an array is constexpr");

  std::vector<int> v;
  assert(std::size(v) == 0);
  assert(std::empty(v));
  v.push_back(7);
  assert(std::size(v) == 1);
  assert(!std::empty(v));
  assert(*std::data(v) == 7);

  const std::vector<int> &cv = v;
  assert(std::size(cv) == 1);
  assert(*std::data(cv) == 7);
  return 0;
}
