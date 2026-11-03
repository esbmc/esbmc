// #9363: k aliased the erased slot, so only the first 2 was removed and this
// wrong post-state was proved.
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_multiset<int> s;
  s.insert(2);
  s.insert(1);
  s.insert(2);
  s.insert(3);
  std::size_t n = s.erase(*s.find(2));
  assert(n == 1 && s.count(2) == 1 && s.size() == 3);
  return 0;
}
