// Reduced from NVIDIA-OpenSMA's nv::spi::buf_to_u32
// (https://github.com/lucasccordeiro/NVIDIA-OpenSMA/tree/vr_sma).
// Idiomatic big-endian byte-deserialiser: under C++20+ each
// `b << k` is well-defined ([expr.shift]/2), so --overflow-check
// must not flag it.
#include <cstdint>

extern "C" unsigned char __VERIFIER_nondet_uchar();

static uint32_t buf_to_u32(const uint8_t *buf)
{
  return (static_cast<uint32_t>(buf[0] << 24)) |
         (static_cast<uint32_t>(buf[1] << 16)) |
         (static_cast<uint32_t>(buf[2] << 8)) |
         (static_cast<uint32_t>(buf[3]));
}

int main()
{
  uint8_t buf[4] = {
    __VERIFIER_nondet_uchar(),
    __VERIFIER_nondet_uchar(),
    __VERIFIER_nondet_uchar(),
    __VERIFIER_nondet_uchar()};

  uint32_t r = buf_to_u32(buf);
  return r == 0 ? 0 : 1;
}
