// Negative counterpart of github_7795: <cfenv> is reachable and a false claim
// about std::fegetround is refuted.
#include <cfenv>
#include <cassert>

int main()
{
  std::fesetround(FE_UPWARD);
  assert(std::fegetround() == FE_TONEAREST);
  return 0;
}
