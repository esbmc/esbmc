#include <set>
#include <memory>
#include <cassert>

int main()
{
  std::multiset<int, std::less<int>, std::allocator<int>> ms;
  ms.insert(5);
  ms.insert(5);
  // A multiset keeps equivalent keys, so this is 2.
  assert(ms.count(5) == 1);
  return 0;
}
