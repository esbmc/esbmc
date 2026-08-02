// Negative companion to github_6318_const_iteration: the const set sums to 6,
// so asserting a wrong total is violated.
#include <cassert>
#include <set>

int sum(const std::set<int> &s)
{
  int t = 0;
  for (auto &x : s)
    t += x;
  return t;
}

int main()
{
  std::set<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(3);
  assert(sum(s) == 99); // wrong on purpose: sum is 6
  return 0;
}
