// Test the ONE bound that actually covers fixed_point::sqrt's machinery:
//
//   sqrt.h:165   "Estimated error bounds: | r - sqrt(x_frac) | < max(1.5 * 2^-11, eps)"
//
// Two things make this the right target and also limit what it proves:
//
//  * It is stated on `sqrt_core`, which takes a NORMALISED argument
//    (0.25 <= x_frac < 1) -- the harness must assume that range or it is
//    testing the function outside its contract.
//  * It bounds r against sqrt(x_frac) BEFORE the rescale
//    `r >>= EXP_ADJUSTMENT - (x_exp >> 1)` that fixed_point::sqrt applies
//    afterwards. So passing here says nothing about sqrt's end-to-end error.
//
// The comparison uses camada's exact mkFXPSqrt on the same symbolic x_frac.
// camada truncates, so the exact root is in [oracle, oracle+1ulp); the bound
// 1.5 * 2^-11 at u0.16 is 1.5 * 2^5 = 48 ulp, so the window is generous and a
// violation here would be a strong result.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned _Fract __ESBMC_fxp_sqrt_ur(unsigned _Fract);
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned short xb = nondet_ushort();
  unsigned _Fract x;
  __ESBMC_bitcast(&x, &xb);

  /* sqrt_core's contract: normalised input, 0.25 <= x_frac < 1.
   * In u0.16 that is raw 16384 .. 65535. */
  __ESBMC_assume(xb >= 16384u);

  unsigned _Fract r =
    fx::sqrt_core<fx::internal::SqrtConfig<unsigned _Fract>>(x);
  unsigned _Fract o = __ESBMC_fxp_sqrt_ur(x);   /* exact, truncated */

  unsigned short rb, ob;
  __ESBMC_bitcast(&rb, &r);
  __ESBMC_bitcast(&ob, &o);

  /* |r - sqrt(x)| < 1.5 * 2^-11. At F=16 one ulp is 2^-16, so the bound is
   * 1.5 * 2^5 = 48 ulp. The exact root is in [ob, ob+1), so |r - root| < 48
   * is implied by |r - ob| <= 48. */
  unsigned int d = (rb > ob) ? (unsigned)(rb - ob) : (unsigned)(ob - rb);
  __ESBMC_assert(d <= 48u, "sqrt_core: |r - sqrt(x_frac)| < 1.5 * 2^-11 (sqrt.h:165)");
  return 0;
}
