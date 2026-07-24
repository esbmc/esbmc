#include <map>
#include <set>
#include <cassert>

int main()
{
  std::map<int, int> m;
  auto r = m.emplace(1, 2);
  assert(r.second == true && r.first->second == 2);
  m.emplace(3, 4);
  assert(m[1] == 2 && m[3] == 4);
  // emplace of an existing key does not overwrite
  auto r2 = m.emplace(1, 99);
  assert(r2.second == false && m[1] == 2);

  std::set<int> s;
  s.emplace(5);
  auto sr = s.emplace(5); // duplicate, not inserted
  assert(s.size() == 1 && sr.second == false && s.count(5) == 1);

  std::multiset<int> ms;
  ms.emplace(7);
  ms.emplace(7); // duplicates allowed
  assert(ms.count(7) == 2);

  std::multimap<int, int> mm;
  mm.emplace(1, 10);
  mm.emplace(1, 20);
  assert(mm.count(1) == 2);

  return 0;
}
