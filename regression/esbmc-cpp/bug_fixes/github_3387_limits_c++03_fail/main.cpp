#include <limits>
#include <cassert>
#include <climits>

// Negative counterpart of github_3387_limits_c++03: confirms the C++03
// numeric_limits members carry their real values rather than being vacuously
// true, by asserting a wrong one.

int main()
{
  assert(std::numeric_limits<int>::digits == 30);
  assert(std::numeric_limits<int>::max() == INT_MAX);
  return 0;
}
