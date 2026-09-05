// sqrtulr (u0.32) against its documented bound, WITHOUT the camada oracle.
//
// The oracle route does not scale here: validating mkFXPSqrt at u0.32 requires
// bracketing a 32-bit symbolic root with 128-bit products, and neither
// bitwuzla nor z3 discharged it (>50 min and >10 min respectively). But the
// oracle is not needed to check libc's claim -- the bound is stated directly
// on the result, exactly as for isqrt:
//
//   |r - true_root| < 1 ulp   <=>   (rb-1)^2 < raw_x * 2^F < (rb+1)^2
//
// This is the same shape that proved uksqrtui in 0.7s, so it should scale.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int xb = nondet_uint();
  unsigned long _Fract x;
  __ESBMC_bitcast(&x, &xb);

  unsigned long _Fract r = fx::sqrt(x);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t xs = (__uint128_t)xb << 32;
  __uint128_t up = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);

  __ESBMC_assert(xs < up, "sqrtulr: true root < r + 1 ulp");

  if (rb > 0)
  {
    __uint128_t dn = ((__uint128_t)rb - 1) * ((__uint128_t)rb - 1);
    __ESBMC_assert(dn < xs, "sqrtulr: true root > r - 1 ulp");
  }
  return 0;
}
