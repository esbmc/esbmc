// Where fixed_point::sqrt's error actually comes from, and why no documented
// bound covers it.
//
// sqrt.h has bounds on sqrt_core (line 165), isqrt (212) and isqrt_fast (237).
// fixed_point::sqrt itself -- which sqrtuhr/sqrtur/sqrtulr/sqrtuhk/sqrtuk all
// call -- has NO accuracy comment; the lines above its declaration are a TODO
// about Newton's method without division.
//
// sqrt_core's bound is proved to hold (harness_sqrt_core_bound.cpp). What sqrt
// adds is normalise -> sqrt_core -> rescale:
//
//     r >>= EXP_ADJUSTMENT - (x_exp >> 1);      // sqrt.h:205, truncating
//
// MEASURED RESULT, and it refutes the obvious theory. At x_raw = 63211 both
// sqrt_core and the full sqrt return 64361 against an exact root of 64363.0025:
// the rescale is a no-op there, and the 2.0 ulp error is already inside
// sqrt_core -- comfortably within its own 48-ulp bound at u0.16.
//
// So the gap is not a rescale bug. It is that sqrt_core's bound
// (1.5 * 2^-11, i.e. 48 ulp at 16 fraction bits) is far looser than one ulp,
// fixed_point::sqrt inherits it unchanged, and nothing in sqrt.h narrows it.
// A caller reading "sqrt" on a u0.16 type gets up to ~48 ulp with no comment
// saying so. That is a documentation gap, not an arithmetic defect.
//
// The assertion states what a reasonable caller would assume from a function
// named sqrt returning the same format: the result is within one ulp of the
// exact root. It is labelled as an assumed contract, not a quoted one.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned _Fract __ESBMC_fxp_sqrt_ur(unsigned _Fract);
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned short xb = nondet_ushort();
  unsigned _Fract x;
  __ESBMC_bitcast(&x, &xb);

  unsigned _Fract l = fx::sqrt(x);              /* full pipeline */
  unsigned _Fract o = __ESBMC_fxp_sqrt_ur(x);   /* exact, truncated */

  unsigned short lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  /* NOT a quoted bound: fixed_point::sqrt documents none. This is the contract
   * a caller would infer from the signature -- within one ulp of the root. */
  __ESBMC_assert((unsigned)lb + 1 >= (unsigned)ob,
                 "sqrt: within 1 ulp of the exact root (INFERRED, not documented)");
  return 0;
}
