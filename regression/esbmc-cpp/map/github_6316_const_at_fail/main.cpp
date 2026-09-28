// Negative companion to github_6316_const_at: at returns the stored value (10),
// so asserting a wrong value is violated.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  const std::map<int, int> &cm = m;
  assert(cm.at(1) == 42); // wrong on purpose: value is 10
  return 0;
}
