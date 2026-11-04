// #9357: an iterator into a multimap survives erase(k) of another key.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_multimap<int, int> m;
  m.insert({1, 10});
  m.insert({2, 20});
  m.insert({1, 11});
  m.insert({3, 30});
  auto mi = m.find(2);
  m.erase(1);
  assert(mi->first != 2);
  return 0;
}
