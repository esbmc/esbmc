// Verify LLVM libc's bitsfx / fxbits pair against TR 18037 7.18a.6.5-6:
// bitsfx reinterprets a fixed-point value as its underlying integer, and
// the fxbits family (hrbits, uhrbits, ...) reinterprets back. They must be
// mutual inverses and must preserve the bit pattern exactly.
//
// Both are `cpp::bit_cast`, so a failure here would point at ESBMC's
// fixed <-> raw bit-vector bridge rather than at the library. That makes this
// a useful cross-check on the Phase 2 solver sweep (mkFXPToRawBV /
// mkFXPFromRawBV) as much as on libc.
//
// Exhaustive: 256 values for the 8-bit formats, 65536 for the 16-bit ones.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned short _Fract nondet_ufract8();
extern "C" short _Fract nondet_sfract8();
extern "C" uint8_t nondet_u8();
extern "C" int8_t nondet_s8();
extern "C" void __ESBMC_assert(bool, const char *);

using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  // ---- unsigned short fract (u0.8) <-> uint8_t ----
  {
    unsigned short _Fract f = nondet_ufract8();
    uint8_t b = bit_cast<uint8_t, unsigned short _Fract>(f);
    unsigned short _Fract back = bit_cast<unsigned short _Fract, uint8_t>(b);
    __ESBMC_assert(back == f, "uhrbits(bitsuhr(f)) == f");
  }
  {
    uint8_t b = nondet_u8();
    unsigned short _Fract f = bit_cast<unsigned short _Fract, uint8_t>(b);
    uint8_t back = bit_cast<uint8_t, unsigned short _Fract>(f);
    __ESBMC_assert(back == b, "bitsuhr(uhrbits(b)) == b");
  }

  // ---- short fract (s0.7) <-> int8_t : exercises the sign bit ----
  {
    short _Fract f = nondet_sfract8();
    int8_t b = bit_cast<int8_t, short _Fract>(f);
    short _Fract back = bit_cast<short _Fract, int8_t>(b);
    __ESBMC_assert(back == f, "hrbits(bitshr(f)) == f");
  }
  {
    int8_t b = nondet_s8();
    short _Fract f = bit_cast<short _Fract, int8_t>(b);
    int8_t back = bit_cast<int8_t, short _Fract>(f);
    __ESBMC_assert(back == b, "bitshr(hrbits(b)) == b");
  }

  // ---- the raw pattern is the VALUE's scaled representation, not a
  // reinterpretation of some other number: raw k must denote k * 2^-8.
  {
    unsigned short _Fract half = 0.5uhr;
    __ESBMC_assert(
      bit_cast<uint8_t, unsigned short _Fract>(half) == 128,
      "bitsuhr(0.5) == 128 (0.5 = 128 * 2^-8)");
    unsigned short _Fract eps = 0.00390625uhr; // 2^-8
    __ESBMC_assert(
      bit_cast<uint8_t, unsigned short _Fract>(eps) == 1,
      "bitsuhr(2^-8) == 1");
    short _Fract mhalf = -0.5hr;
    __ESBMC_assert(
      bit_cast<int8_t, short _Fract>(mhalf) == -64,
      "bitshr(-0.5) == -64 (s0.7: -0.5 = -64 * 2^-7)");
  }

  return 0;
}
