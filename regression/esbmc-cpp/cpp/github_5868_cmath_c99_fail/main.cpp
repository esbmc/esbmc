// Negative counterpart of github_5868_cmath_c99: the newly visible std::
// overloads forward to the real models rather than returning an unconstrained
// value, so a wrong claim about them is refuted.
#include <cmath>
#include <cassert>

int main()
{
  assert(std::fmin(1.0, 2.0) == 2.0);
  return 0;
}
