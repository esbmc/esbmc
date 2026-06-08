// Same expression as cpp20_shl_uint8_promoted_defined but compiled at
// `--std c++17`. Under C++17, the [expr.shift]/2 rule required
// E1 * 2^E2 to be representable in the *result* type; for byte = 128,
// 128 << 24 = 2^31 is not representable as int -> UB. The skip in
// goto_check.cpp must therefore NOT fire under pre-C++20 standards;
// this regression confirms standard-discrimination.
#include <cstdint>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
