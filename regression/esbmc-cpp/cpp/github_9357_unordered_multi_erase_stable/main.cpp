// #9357: erase(k) of a multimap/multiset invalidates only the erased elements.
#include <cassert>
#include <unordered_map>
#include <unordered_set>

int main()
{
  std::unordered_multimap<int, int> m;
  m.insert({1, 10});
  m.insert({2, 20});
  m.insert({1, 11});
  m.insert({3, 30});
  auto mi = m.find(2);
  assert(m.erase(1) == 2);
  assert(mi->first == 2 && mi->second == 20);
  assert(m.size() == 2 && m.count(1) == 0);
  m.insert({4, 40});
  assert(m.size() == 3 && m.find(4)->second == 40);

  std::unordered_multiset<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(1);
  s.insert(3);
  auto si = s.find(2);
  assert(s.erase(1) == 2);
  assert(*si == 2);
  assert(s.size() == 2 && s.count(1) == 0);
  s.insert(4);
  assert(s.size() == 3 && s.count(4) == 1);
  return 0;
}
