// #9363: erase(k) with k referring to a stored element must still remove
// every equivalent element.
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
  assert(n == 2 && s.count(2) == 0 && s.size() == 2);
  return 0;
}
