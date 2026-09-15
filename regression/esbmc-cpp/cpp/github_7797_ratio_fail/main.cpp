// Negative counterpart of github_7797_ratio: 1/1000 + 1/1000000 is
// 1001/1000000.
#include <ratio>
#include <cassert>

using MilliPlusMicro = std::ratio_add<std::milli, std::micro>;

int main()
{
  assert(MilliPlusMicro::num == 1000);
  return 0;
}
