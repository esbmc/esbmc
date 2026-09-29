// isqrt against its documented bound: "Absolute errors < 2^(-fraction length)",
// i.e. strictly under one ulp -- in EITHER direction. libc does not claim
// truncation, so the property must permit rounding up as well as down.
//
// The earlier version of this harness asserted rb <= floor(true root), which
// forbids rounding up and produced a false positive at n = 65534, where
// sqrt(65534)*256 = 65534.99999 and libc correctly returns 65535 (0.0000038
// ulp off). That was my bracket, not a defect.
//
// Stated with no reference computed here, on raw integers at full width:
//   |r - true_root| < 1 ulp   <=>   (rb-1)^2 < n * 2^(2F) < (rb+1)^2
// The strict inequalities are what "< 1 ulp" means; equality would be exactly
// one ulp off and is excluded by the claim.
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
  unsigned _Accum r = fx::isqrt(n);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t ns = (__uint128_t)n << 32;

  /* Exclude saturation. Where the true root exceeds the format maximum,
   * clamping is correct and the error bound does not apply -- tested on the
   * TRUE root, not on the returned value: at u8.8, n = 65535 has root
   * 255.99805 above MAX 255.99609, and libc returns 65534 there rather than
   * MAX, so a guard on the returned raw would have missed it. */
  {
    __uint128_t mx = (__uint128_t)4294967295u * 4294967295u;
    if (ns > mx)
      return 0;
  }
  __uint128_t up = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);

  /* upper side: the true root is below rb+1 ulp */
  __ESBMC_assert(ns < up, "isqrt: true root < r + 1 ulp");

  /* lower side: the true root is above rb-1 ulp. Guard rb == 0, where
   * there is no rb-1 and the claim is vacuous. */
  if (rb > 0)
  {
    __uint128_t dn = ((__uint128_t)rb - 1) * ((__uint128_t)rb - 1);
    __ESBMC_assert(dn < ns, "isqrt: true root > r - 1 ulp");
  }
  return 0;
}
