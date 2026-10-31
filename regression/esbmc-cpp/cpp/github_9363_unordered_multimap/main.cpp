// #9363: erase(k) with k referring to a stored key must still remove every
// equivalent element.
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
  assert(n == 2 && m.count(2) == 0 && m.size() == 2);
  return 0;
}
