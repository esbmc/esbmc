// `uint8_t b; b << 24` is the firmware byte-deserialiser idiom that
// motivated #4201. After integer promotion the operand has type `int`
// and value in [0, 255] -- statically non-negative. Under C++20
// [expr.shift]/2 the shift is the unique value congruent to b * 2^24
// modulo 2^32, well-defined.
#include <cstdint>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
