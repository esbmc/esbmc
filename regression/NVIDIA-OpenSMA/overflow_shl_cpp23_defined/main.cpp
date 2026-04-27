#include <cstdint>

extern "C" unsigned char __VERIFIER_nondet_uchar();

int main()
{
  uint8_t b = __VERIFIER_nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
