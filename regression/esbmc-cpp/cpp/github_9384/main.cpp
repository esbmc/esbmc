// github #9384: multiset::erase(k) erases every element equal to k and
// returns how many it erased ([associative.reqmts] Table 90).
#include <cassert>
#include <set>

int main()
{
  std::multiset<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(1);
  s.insert(3);
  s.insert(1);
  size_t n = s.erase(1);
  assert(n == 3 && s.size() == 2);
  assert(s.count(1) == 0 && *s.begin() == 2);
  assert(s.erase(7) == 0 && s.size() == 2);
  return 0;
}
