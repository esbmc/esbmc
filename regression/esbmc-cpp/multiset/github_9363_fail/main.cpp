// #9363: x aliased the erased slot and was overwritten by a 3, so a 3 was
// erased too and this wrong post-state was proved.
#include <cassert>
#include <set>

int main()
{
  std::multiset<int> s;
  s.insert(2);
  s.insert(2);
  s.insert(3);
  s.insert(3);
  std::size_t n = s.erase(*s.find(2));
  assert(n == 3 && s.count(3) == 1 && s.size() == 1);
  return 0;
}
