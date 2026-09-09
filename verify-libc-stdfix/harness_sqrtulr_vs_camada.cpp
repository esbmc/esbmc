// sqrtulr against camada's mkFXPSqrt on the same symbolic input.
//
//   x        : symbolic unsigned long _Fract -- every value at once
//   libc     : fixed_point::sqrt(x)
//   oracle   : __ESBMC_fxp_sqrt_ulr(x)   -- camada mkFXPSqrt, exact
//
// sqrt.h:211-212 claims "Absolute errors < 2^(-fraction length)" = < 1 ulp.
// mkFXPSqrt truncates toward zero, so the true root is in [oracle, oracle+1ulp)
// and an error under 1 ulp forces the result into {oracle, oracle+1ulp}.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"

extern "C" unsigned long _Fract __ESBMC_fxp_sqrt_ulr(unsigned long _Fract);
extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int xb = nondet_uint();
  unsigned long _Fract x;
  __ESBMC_bitcast(&x, &xb);

  unsigned long _Fract l = fx::sqrt(x);
  unsigned long _Fract o = __ESBMC_fxp_sqrt_ulr(x);

  unsigned int lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb >= ob, "sqrtulr: not below the exact root");
  __ESBMC_assert((unsigned long long)lb <= (unsigned long long)ob + 1,
                 "sqrtulr: within 1 ulp of the exact root");
  return 0;
}
