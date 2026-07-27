// <limits> used FLT_DECIMAL_DIG / FLT_TRUE_MIN and friends, which <float.h>
// only exposes from C11 / C++17 on, so the header failed to parse under
// --std c++11 and c++14 (github #3387).  The values asserted here match the
// host standard library.
#include <limits>
#include <cassert>

int main()
{
  assert(std::numeric_limits<float>::max_digits10 == 9);
  assert(std::numeric_limits<double>::max_digits10 == 17);

  assert(std::numeric_limits<float>::denorm_min() > 0.0F);
  assert(
    std::numeric_limits<float>::denorm_min() <
    std::numeric_limits<float>::min());
  assert(std::numeric_limits<double>::denorm_min() > 0.0);
  assert(
    std::numeric_limits<double>::denorm_min() <
    std::numeric_limits<double>::min());

  assert(std::numeric_limits<float>::digits == 24);
  assert(std::numeric_limits<double>::digits == 53);
  assert(std::numeric_limits<int>::is_specialized);

  return 0;
}
