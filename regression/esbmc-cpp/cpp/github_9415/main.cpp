#include <cassert>
#include <climits>
#include <string>

int main()
{
  assert(std::to_string(4294967296ULL) == "4294967296");
  assert(std::to_string(-5000000000LL) == "-5000000000");
  assert(std::to_string(LLONG_MIN) == "-9223372036854775808");
  assert(std::to_string(ULLONG_MAX) == "18446744073709551615");
  assert(std::to_string(INT_MIN) == "-2147483648");
  assert(std::to_string(4000000000U) == "4000000000");
  assert(std::to_string(-3000000000L) == "-3000000000");
  assert(std::to_string(3000000000UL) == "3000000000");
  assert(std::to_string(0) == "0");
  assert(std::to_string(-7) == "-7");
  return 0;
}
