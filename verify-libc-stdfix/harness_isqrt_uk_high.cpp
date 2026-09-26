// isqrt (uhksqrtus / uksqrtui) against an exact reference, in the solver.
//
// isqrt is where the violated claim actually SITS: "Integer square root -
// Accurate version: Absolute errors < 2^(-fraction length)" (sqrt.h:211-212).
// uhksqrtus and uksqrtui call this, so the bound is violated on its own entry
// points, not merely inherited from the neighbouring fixed_point::sqrt.
//
// The shape differs from the fract case: the argument is an INTEGER and the
// result is an _Accum, so the reference cannot be mkFXPSqrt of the same
// format. Instead the property is stated directly on the result, exactly as
// for the oracle's own validation and with no expected value computed here:
//
//   r = isqrt(n)  with r in u8.8 (F = 8)
//   error < 1 ulp  <=>  (r - 1ulp)^2 < n <= (r + 1ulp)^2
//
// In raw integer terms, with r_raw = r * 2^F and n scaled by 2^(2F):
//   (r_raw - 1)^2 < n * 2^(2F) < (r_raw + 1)^2
// evaluated at full width so no fixed-point rounding enters the bracket --
// the mistake that made two earlier bracket attempts fail.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int n = nondet_uint();

  // uhksqrtus: unsigned short -> u8.8, so F = 8.
  unsigned _Accum r = fx::isqrt(n);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  // exact integer bracket. rb is already scaled by 2^F (F = 8), so rb^2
  // carries scale 2^16 and n must be scaled to match:
  //   rb^2 <= n * 2^32 < (rb+1)^2
  __uint128_t nn = (__uint128_t)n << 32;
  __uint128_t lo = (__uint128_t)rb * rb;
  __uint128_t hi = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);

  __ESBMC_assert(nn < hi, "isqrt: n < (r+1ulp)^2 (within 1 ulp of the root)");
  return 0;
}
