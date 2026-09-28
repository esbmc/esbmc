// Negative counterpart of github_3387_limits_cxx11: max_digits10 for float is
// 9, not 8, so the program is really verified rather than passing vacuously.
#include <limits>
#include <cassert>

int main()
{
  assert(std::numeric_limits<float>::max_digits10 == 8);
  return 0;
}
