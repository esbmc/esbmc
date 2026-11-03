// github #9403: multimap::find on an absent key returns end() and does not
// insert ([associative.reqmts] Table 90).
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert(std::make_pair(3, 30));
  m.insert(std::make_pair(5, 50));
  m.insert(std::make_pair(5, 51));
  assert(m.find(1) == m.end());
  assert(m.find(4) == m.end());
  assert(m.find(9) == m.end());
  assert(m.size() == 3);
  assert(m.count(1) == 0 && m.count(9) == 0);
  assert(m.find(5) != m.end() && m.find(5)->first == 5);
  assert(m.find(3)->second == 30);
  return 0;
}
