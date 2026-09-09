// BUG 3 of 3, and the one with a different character: at s32.31 / u32.32,
// divi's intermediate format IS the result format, so there is no headroom.
//
// fx_bits.h:266-267 casts the raw int64 `res64` to `long accum` as a VALUE
// and divides by 2^F. For a narrow XType that is harmless because long accum
// (s32.31, range +-2^32) has headroom over the target. For XType == long
// _Accum the intermediate and the target are the same format, and any res64
// beyond +-2^32 is lost: measured, 2^33 and above convert to 0.0, which is
// why divilk(-64,-1) returns 0 where the answer is 64.0.
//
// IMPORTANT distinction from bugs 1 and 2: TR 18037 4.1.3 makes conversion of
// a value not representable in an UNSATURATED fixed-point type UNDEFINED. So
// the compiler wrapping to zero is not itself wrong -- the defect is that
// divi depends on that conversion for inputs it accepts. That is a weaker
// claim than the sign law, and it is stated as such.
//
// The property asserted here needs no accuracy contract either: a quotient
// whose exact value is representable in the result format must not come back
// as zero.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  int n = nondet_int(), d = nondet_int();
  __ESBMC_assume(d == 1 || d == -1); // exact, so no rounding question at all
  __ESBMC_assume(n >= -64 && n <= 64 && n != 0);

  long _Accum r = fx::divi<long _Accum>(n, d);

  // |n/d| is between 1 and 64, comfortably inside s32.31's +-2^32 range, so
  // the result is exactly representable and cannot legitimately be zero.
  __ESBMC_assert(
    bit_cast<int64_t, long _Accum>(r) != 0,
    "divilk: an exactly-representable nonzero quotient is not zero");

  return 0;
}
