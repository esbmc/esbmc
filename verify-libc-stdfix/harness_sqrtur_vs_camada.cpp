// sqrtur against camada's mkFXPSqrt on the same symbolic input.
//
//   x        : symbolic unsigned _Fract -- every value at once
//   libc     : fixed_point::sqrt(x)
//   oracle   : __ESBMC_fxp_sqrt_ur(x)   -- camada mkFXPSqrt, exact
//
// sqrt.h:211-212 claims "Absolute errors < 2^(-fraction length)" = < 1 ulp.
// mkFXPSqrt truncates toward zero, so the true root is in [oracle, oracle+1ulp)
// and an error under 1 ulp forces the result into {oracle, oracle+1ulp}.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"

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

  unsigned _Fract l = fx::sqrt(x);
  unsigned _Fract o = __ESBMC_fxp_sqrt_ur(x);

  unsigned short lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb >= ob, "sqrtur: not below the exact root");
  __ESBMC_assert((unsigned long long)lb <= (unsigned long long)ob + 1,
                 "sqrtur: within 1 ulp of the exact root");
  return 0;
}
