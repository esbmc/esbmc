#include <map>
#include <cassert>

int lookup(const std::map<int, int> &m)
{
  return m.find(1)->second;
}

int main()
{
  std::map<int, int> m;
  m[1] = 42;
  assert(lookup(m) == 7); // must fail: const find returns the stored 42
  return 0;
}
