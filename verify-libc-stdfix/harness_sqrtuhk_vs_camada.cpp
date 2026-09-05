// sqrtuhk against camada's mkFXPSqrt on the same symbolic input.
//
// sqrtuhk.cpp calls fixed_point::sqrt(x), so the claim is sqrt.h:211-212's
// "Absolute errors < 2^(-fraction length)" -- under 1 ulp.
//
// mkFXPSqrt is the exact root truncated toward zero, so the true root lies in
// [oracle, oracle+1ulp). An implementation with absolute error under 1 ulp must
// land in {oracle, oracle+1ulp}; anything else is more than an ulp out.
//
// Both operand and result are unsigned short _Accum, so the oracle applies directly --
// no rescaling, unlike the isqrt_fast entry points which take an integer.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"

extern "C" unsigned short _Accum __ESBMC_fxp_sqrt_uhk(unsigned short _Accum);
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned short xb = nondet_ushort();
  unsigned short _Accum x;
  __ESBMC_bitcast(&x, &xb);

  unsigned short _Accum l = fx::sqrt(x);                 // libc
  unsigned short _Accum o = __ESBMC_fxp_sqrt_uhk(x);   // camada, exact

  unsigned short lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb >= ob, "sqrtuhk: not below the exact root");
  __ESBMC_assert((unsigned long long)lb <= (unsigned long long)ob + 1,
                 "sqrtuhk: within 1 ulp of the exact root (sqrt.h:211)");
  return 0;
}
