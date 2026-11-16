// Same expression as cpp20_shl_uint8_promoted_defined but compiled at
// `--std c++17`. C++14/17 [expr.shift]/2 (CWG 1457) defines E1 << E2 for a
// non-negative E1 when E1 * 2^E2 fits the corresponding *unsigned* type:
// 128 << 24 = 2^31 fits unsigned int, so it is defined, not UB. The overflow
// past that bound is shl_past_unsigned_cpp17_fail.
#include <cstdint>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
