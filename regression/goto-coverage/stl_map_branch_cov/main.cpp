#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;

  int key = nondet_int();
  int result;
  if (m.count(key) > 0)
    result = m[key];
  else
    result = -1;

  // 3-way OR is load-bearing for the cond_cov variant's pinned
  // Total Conditions count; do not simplify.
  assert(result == 10 || result == 20 || result == -1);
  return 0;
}
