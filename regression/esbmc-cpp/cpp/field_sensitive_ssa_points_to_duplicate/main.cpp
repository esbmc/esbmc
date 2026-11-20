// The map's iterator holds pointers into m's arrays, and Andersen lists m
// twice in their points-to set.
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert(std::make_pair(3, 30));
  m.insert(std::make_pair(5, 50));
  assert(m.find(1) == m.end());
  assert(m.find(5)->second == 50);
  return 0;
}
