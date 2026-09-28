#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[3] = 30;

  // lower_bound(2) is the element with key 3, not the end of the map.
  assert(m.lower_bound(2) == m.end());

  return 0;
}
