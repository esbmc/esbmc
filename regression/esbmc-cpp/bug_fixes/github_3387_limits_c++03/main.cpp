#include <limits>
#include <cassert>
#include <climits>
#include <cfloat>

int main()
{
  assert(std::numeric_limits<int>::is_specialized);
  assert(std::numeric_limits<int>::is_signed);
  assert(std::numeric_limits<int>::is_integer);
  assert(std::numeric_limits<int>::digits == 31);
  assert(std::numeric_limits<int>::min() == INT_MIN);
  assert(std::numeric_limits<int>::max() == INT_MAX);
  assert(std::numeric_limits<unsigned>::min() == 0u);
  assert(std::numeric_limits<unsigned>::max() == UINT_MAX);
  assert(!std::numeric_limits<unsigned>::is_signed);
  assert(std::numeric_limits<char>::is_specialized);
  assert(std::numeric_limits<double>::is_iec559);
  assert(std::numeric_limits<double>::has_infinity);
  assert(!std::numeric_limits<double>::is_integer);
  assert(std::numeric_limits<double>::radix == FLT_RADIX);
  assert(std::numeric_limits<double>::max() == DBL_MAX);
  assert(std::numeric_limits<float>::epsilon() == FLT_EPSILON);
  assert(std::numeric_limits<float>::round_style == std::round_to_nearest);
  assert(std::numeric_limits<long>::max() == LONG_MAX);
  assert(std::numeric_limits<short>::min() == SHRT_MIN);
  return 0;
}
