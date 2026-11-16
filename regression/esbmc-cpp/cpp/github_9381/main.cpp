// github #9381: multimap::erase(k) erases every element with key k and
// returns how many it erased ([associative.reqmts] Table 90).
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert({1, 10});
  m.insert({1, 11});
  m.insert({2, 20});
  m.insert({1, 12});
  int n = m.erase(1);
  assert(n == 3 && m.size() == 1);
  assert(m.count(1) == 0 && m.begin()->first == 2 && m.begin()->second == 20);
  assert(m.erase(7) == 0 && m.size() == 1);
  return 0;
}
