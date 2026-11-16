// #9363: erase(k) with k referring to a stored element must remove only the
// elements equivalent to it.
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
  assert(n == 2 && s.count(3) == 2 && s.size() == 2);
  return 0;
}
