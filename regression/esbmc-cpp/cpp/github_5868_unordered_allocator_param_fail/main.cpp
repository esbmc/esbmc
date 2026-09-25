#include <unordered_set>
#include <memory>
#include <functional>
#include <cassert>

int main()
{
  std::unordered_set<
    int, std::hash<int>, std::equal_to<int>, std::allocator<int>> s;
  s.insert(4);
  // A set holds one copy of an equivalent key.
  assert(s.count(4) == 2);
  return 0;
}
