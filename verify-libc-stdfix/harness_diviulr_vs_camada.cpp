// diviulr against camada's mkFXPDiv.
//
// diviulr calls divifx<unsigned int, unsigned long _Fract>(n, d). The camada-side division is done in
// unsigned long _Accum, which has the same 32 fraction bits as unsigned long _Fract -- so widening the
// divisor preserves its value exactly -- plus integer bits for the dividend and
// the quotient.
//
// mkFXPDiv is format-generic (no allowlist, unlike mkFXPExp), so the only real
// constraints are the C type ranges: n and n/d must fit the format the division
// happens in.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_n();
extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int n = nondet_n();
  unsigned int db = nondet_uint();
  __ESBMC_assume(db != 0);
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db >= (((unsigned int)1) << 28));

  unsigned long _Fract d;
  __ESBMC_bitcast(&d, &db);

  unsigned int q_libc = fx::divifx<unsigned int, unsigned long _Fract>(n, d);

  unsigned long _Accum dw = (unsigned long _Accum)d;
  unsigned long _Accum nw = (unsigned long _Accum)n;
  unsigned long _Accum qw = nw / dw;
  unsigned int q_camada = (unsigned int)qw;

  __ESBMC_assert(q_libc == q_camada,
                 "diviulr: divifx agrees with camada's division");
  return 0;
}
