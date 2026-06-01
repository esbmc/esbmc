// Legacy `--std c++98` numerically parses as 98 in is_cxx20_or_later;
// the year-range bound (year >= 20 && year <= 50) must reject it so the
// pre-C++20 overflow rule still applies. Confirms the predicate does
// not silently mistake the legacy spelling for C++20+.
#include <stdint.h>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
