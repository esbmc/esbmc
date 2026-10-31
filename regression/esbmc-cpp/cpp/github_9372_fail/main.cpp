// github #9372: erasing an absent key from a two-element map returns 0, not
// the map's size.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  std::size_t r = m.erase(7);
  assert(r == 2);
  return 0;
}
