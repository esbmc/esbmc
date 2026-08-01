// Negative companion to github_6318_map_const_iteration: the const map's values
// sum to 30, so asserting a wrong total is violated.
#include <cassert>
#include <map>

int sum_values(const std::map<int, int> &m)
{
  int t = 0;
  for (auto &kv : m)
    t += kv.second;
  return t;
}

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  assert(sum_values(m) == 99); // wrong on purpose: sum is 30
  return 0;
}
