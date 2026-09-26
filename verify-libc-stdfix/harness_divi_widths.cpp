// BUG 1 is not width-specific: the same double negation is proved at
// s0.15 and s0.31, not only at the s0.7 of harness_divi_bug1.cpp.
//
// The _Accum types escape it for a different reason worth stating: they
// have integer bits, so these quotients stay in range and never reach the
// saturation rail where the flipped sign becomes visible as MIN-vs-MAX.
// The double negation still happens there; it just shows up as a wrong
// value rather than a wrong rail.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---- BUG 1 again at s0.15 and s0.31, to show it is not width-specific ---- */
  {
    int n = nondet_int(), d = nondet_int();
    __ESBMC_assume(d == -2 || d == -4);
    __ESBMC_assume(n < 0 && n > -128);
    __ESBMC_assume(-n >= -d);
    __ESBMC_assert(
      bit_cast<int16_t, _Fract>(fx::divi<_Fract>(n, d)) > 0,
      "divir: neg/neg is positive");
    __ESBMC_assert(
      bit_cast<int32_t, long _Fract>(fx::divi<long _Fract>(n, d)) > 0,
      "divilr: neg/neg is positive");
  }

  return 0;
}
