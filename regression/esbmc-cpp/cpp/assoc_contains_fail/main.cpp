#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m[1] = 2;
  assert(m.contains(9)); // key 9 is absent
  return 0;
}
