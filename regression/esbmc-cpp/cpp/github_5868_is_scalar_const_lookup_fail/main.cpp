#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m[2] = 20;
  const std::map<int, int> &cm = m;
  // upper_bound(2) is past the last element, so it is end().
  assert(cm.upper_bound(2) != cm.end());
  return 0;
}
