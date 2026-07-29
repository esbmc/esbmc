// Negative counterpart of github_5868_transparent_functors: std::less<> really
// compares at the operands' own types, so the truncating answer std::less<int>
// used to give (0.5 < 0.7 == false) is now refuted.
#include <functional>
#include <cassert>

int main()
{
  assert(std::less<>()(0.5, 0.7) == false);
  return 0;
}
