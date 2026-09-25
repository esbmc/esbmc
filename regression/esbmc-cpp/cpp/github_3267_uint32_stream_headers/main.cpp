// libstdc++ 11 (-std=c++17) makes the fixed-width typedefs visible through
// each of these; ESBMC did not, so unmodified sources failed to parse
// (github #3267).
#include <iostream>
#include <ostream>
#include <istream>
#include <streambuf>
#include <iomanip>
#include <iterator>
#include <memory>
#include <cassert>

int main()
{
  uint32_t n = 4000000000u;
  assert(n > 2147483647u);
  assert(sizeof(uint32_t) == 4);

  std::uint32_t qn = n;
  std::int64_t big = INT64_C(-5000000000);
  std::size_t sz = 3;
  uint_least16_t least = 7;
  assert(qn == n);
  assert(big < -2147483648LL);
  assert(sz + least == 10);
  assert(UINT32_MAX == 4294967295u);
  assert(sizeof(intmax_t) >= sizeof(int32_t));
  return 0;
}
