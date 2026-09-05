// diviur against camada's mkFXPDiv.
//
// diviur calls divifx<unsigned int, unsigned _Fract>(n, d). The camada-side division is done in
// unsigned _Accum, which has the same 16 fraction bits as unsigned _Fract -- so widening the
// divisor preserves its value exactly -- plus integer bits for the dividend and
// the quotient.
//
// mkFXPDiv is format-generic (no allowlist, unlike mkFXPExp), so the only real
// constraints are the C type ranges: n and n/d must fit the format the division
// happens in.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_n();
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int n = nondet_n();
  unsigned short db = nondet_ushort();
  __ESBMC_assume(db != 0);
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db >= (((unsigned short)1) << 11));

  unsigned _Fract d;
  __ESBMC_bitcast(&d, &db);

  unsigned int q_libc = fx::divifx<unsigned int, unsigned _Fract>(n, d);

  unsigned _Accum dw = (unsigned _Accum)d;
  unsigned _Accum nw = (unsigned _Accum)n;
  unsigned _Accum qw = nw / dw;
  unsigned int q_camada = (unsigned int)qw;

  __ESBMC_assert(q_libc == q_camada,
                 "diviur: divifx agrees with camada's division");
  return 0;
}
