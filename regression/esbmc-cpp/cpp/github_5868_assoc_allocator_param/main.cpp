#include <map>
#include <set>
#include <memory>
#include <cassert>

int main()
{
  std::map<int, int, std::less<int>, std::allocator<std::pair<const int, int>>> m;
  m[1] = 10;
  assert(m[1] == 10);

  std::set<int, std::less<int>, std::allocator<int>> s;
  s.insert(4);
  assert(s.count(4) == 1);

  std::multiset<int, std::less<int>, std::allocator<int>> ms;
  ms.insert(5);
  ms.insert(5);
  assert(ms.count(5) == 2);

  std::multimap<int, int, std::less<int>, std::allocator<std::pair<const int, int>>> mm;
  mm.insert(std::pair<int, int>(2, 20));
  assert(mm.size() == 1);

  std::map<int, int> dm;
  dm[3] = 30;
  assert(dm[3] == 30);
  return 0;
}
