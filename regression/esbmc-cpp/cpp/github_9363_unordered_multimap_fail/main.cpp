// #9363: k aliased the erased slot, so only the first 2 was removed and this
// wrong post-state was proved.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_multimap<int, int> m;
  m.insert({2, 0});
  m.insert({1, 0});
  m.insert({2, 1});
  m.insert({3, 0});
  std::size_t n = m.erase(m.find(2)->first);
  assert(n == 1 && m.count(2) == 1 && m.size() == 3);
  return 0;
}
