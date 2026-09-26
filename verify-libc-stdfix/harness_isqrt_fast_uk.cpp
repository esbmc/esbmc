// uksqrtui against the claim it actually makes.
//
// uksqrtui.cpp:18 returns fixed_point::isqrt_fast(x), NOT isqrt. sqrt.h states
// different bounds for the two:
//
//   isqrt      (sqrt.h:211)  "Absolute errors < 2^(-fraction length)"
//   isqrt_fast (sqrt.h:236)  "Relative errors < 2^(-fraction length)"
//
// So the property here is the RELATIVE bound, on the function the entry point
// calls. Earlier harnesses asserted the absolute bound on isqrt, which is a
// function uksqrtui does not call and a claim it does not make.
//
// Stated without computing a reference: with r the result scaled by 2^16 and
// the true root sqrt(n), "relative error < 2^-16" is
//
//     |r/2^16 - sqrt(n)| / sqrt(n) < 2^-16
//
// Multiplying through by sqrt(n)*2^16 and squaring to stay in integers:
//
//     (r - 2^16)^2 < n * 2^32 < (r + 2^16)^2      [scaled by (1 -/+ 2^-16)]
//
// i.e. r must lie within a factor (1 +/- 2^-16) of the true root. Products on
// raw integers at full width so no fixed-point rounding enters the bound.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_in();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int n = nondet_in();
  unsigned _Accum r = fx::isqrt_fast(n);   // the function uksqrtui actually calls
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t ns = (__uint128_t)n << 32;

  /* Exclude saturation: where the true root exceeds the format maximum,
   * clamping is correct and no error bound applies. */
  {
    __uint128_t mx = (__uint128_t)((unsigned int)-1) * (__uint128_t)((unsigned int)-1);
    if (ns > mx)
      return 0;
  }

  /* relative bound: r must be within a factor (1 +/- 2^-16) of the true root.
   * Upper side: r < sqrt(n)*2^16 * (1 + 2^-16) = sqrt(n)*2^16 + sqrt(n).
   * Squaring, with both sides scaled by 2^32: */
  __uint128_t up = ((__uint128_t)rb + ((__uint128_t)1 << 16)) *
                   ((__uint128_t)rb + ((__uint128_t)1 << 16));
  __ESBMC_assert(ns < up, "uksqrtui: relative error < 2^-16 (upper side)");

  if (rb > ((__uint128_t)1 << 16))
  {
    __uint128_t dn = ((__uint128_t)rb - ((__uint128_t)1 << 16)) *
                     ((__uint128_t)rb - ((__uint128_t)1 << 16));
    __ESBMC_assert(dn < ns, "uksqrtui: relative error < 2^-16 (lower side)");
  }
  return 0;
}
