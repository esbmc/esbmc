#include <unordered_set>
#include <memory>
#include <functional>
#include <cassert>

int main()
{
  // [unord.multiset.overview]: four parameters.
  std::unordered_multiset<
    int, std::hash<int>, std::equal_to<int>, std::allocator<int>> m;
  m.insert(1);
  m.insert(1);
  assert(m.count(1) == 2);
  assert(m.size() == 2);

  // The defaults must still work.
  std::unordered_multiset<int> d;
  d.insert(5);
  d.insert(5);
  assert(d.count(5) == 2);
  return 0;
}
