#include <map>
#include <set>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[3] = 30;
  assert(m.contains(1) && m.contains(3) && !m.contains(9));

  std::map<int, int> empty;
  assert(!empty.contains(1)); // empty map

  std::multimap<int, int> mm;
  mm.insert(std::make_pair(1, 2));
  mm.insert(std::make_pair(1, 3));
  assert(mm.contains(1) && !mm.contains(9));

  std::set<int> s;
  s.insert(5);
  s.insert(7);
  assert(s.contains(5) && s.contains(7) && !s.contains(9));

  std::multiset<int> ms;
  ms.insert(5);
  ms.insert(5);
  assert(ms.contains(5) && !ms.contains(9));

  return 0;
}
