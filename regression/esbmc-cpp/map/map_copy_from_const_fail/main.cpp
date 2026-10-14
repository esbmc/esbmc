// Anti-vacuity twin of map_copy_from_const: the copy has to carry the source's
// elements, not merely compile.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 7;

  const std::map<int, int> &cref = m;
  std::map<int, int> copy(cref);
  assert(copy[1] == 8);
  return 0;
}
