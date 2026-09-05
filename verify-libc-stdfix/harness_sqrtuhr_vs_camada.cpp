// sqrtuhr against camada's mkFXPSqrt on the same symbolic input.
//
//   x        : symbolic unsigned short _Fract -- every value at once
//   libc     : fixed_point::sqrt(x)
//   oracle   : __ESBMC_fxp_sqrt_uhr(x)   -- camada mkFXPSqrt, exact
//
// sqrt.h:211-212 claims "Absolute errors < 2^(-fraction length)" = < 1 ulp.
// mkFXPSqrt truncates toward zero, so the true root is in [oracle, oracle+1ulp)
// and an error under 1 ulp forces the result into {oracle, oracle+1ulp}.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"

extern "C" unsigned short _Fract __ESBMC_fxp_sqrt_uhr(unsigned short _Fract);
extern "C" unsigned char nondet_uchar();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned char xb = nondet_uchar();
  unsigned short _Fract x;
  __ESBMC_bitcast(&x, &xb);

  unsigned short _Fract l = fx::sqrt(x);
  unsigned short _Fract o = __ESBMC_fxp_sqrt_uhr(x);

  unsigned char lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb >= ob, "sqrtuhr: not below the exact root");
  __ESBMC_assert((unsigned long long)lb <= (unsigned long long)ob + 1,
                 "sqrtuhr: within 1 ulp of the exact root");
  return 0;
}
