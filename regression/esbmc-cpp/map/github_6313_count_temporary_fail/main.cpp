// Negative companion to github_6313_count_temporary: key 1 is present, so
// asserting count(1) == 0 is violated.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  assert(m.count(1) == 0); // wrong on purpose: key 1 is present
  return 0;
}
