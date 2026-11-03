// github #9403: multimap::find on an absent key must not insert it.
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert(std::make_pair(3, 30));
  m.find(1);
  assert(m.size() == 2);
  return 0;
}
