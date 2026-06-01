// Same OpenSMA-style byte deserialiser as cpp20_shl_uint8_promoted_defined,
// compiled with --std gnu++20. Confirms the gnu++ prefix is recognised by
// is_cxx20_or_later() and the skip applies.
#include <cstdint>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
