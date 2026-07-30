// Non-vacuity guard: upper_bound really returns the first element greater than
// the value, so the old (backwards-scan) answer must FAIL.
#include <algorithm>
#include <cassert>

int main()
{
  int v[6] = {1, 2, 3, 4, 5, 6};
  assert(std::upper_bound(v, v + 6, 3) == v + 5); // the pre-fix answer
  return 0;
}
