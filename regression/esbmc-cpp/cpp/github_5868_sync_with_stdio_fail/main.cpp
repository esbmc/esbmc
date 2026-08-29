// Negative counterpart of github_5868_sync_with_stdio: the returned state is
// the real previous one, not a nondeterministic value, so a wrong claim about
// it is refuted.
#include <iostream>
#include <cassert>

int main()
{
  std::ios_base::sync_with_stdio(false);
  assert(std::ios_base::sync_with_stdio(true) == true);
  return 0;
}
