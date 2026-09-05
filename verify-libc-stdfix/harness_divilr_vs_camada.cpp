// divilr against camada's mkFXPDiv.
//
// divilr calls divifx<int, long _Fract>(n, d). The camada-side division is done in
// long _Accum, which has the same 31 fraction bits as long _Fract -- so widening the
// divisor preserves its value exactly -- plus integer bits for the dividend and
// the quotient.
//
// mkFXPDiv is format-generic (no allowlist, unlike mkFXPExp), so the only real
// constraints are the C type ranges: n and n/d must fit the format the division
// happens in.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_n();
extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  int n = nondet_n();
  int db = nondet_int();
  __ESBMC_assume(db != 0);
  __ESBMC_assume(db > 0);
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db >= (((int)1) << 27));

  long _Fract d;
  __ESBMC_bitcast(&d, &db);

  int q_libc = fx::divifx<int, long _Fract>(n, d);

  long _Accum dw = (long _Accum)d;
  long _Accum nw = (long _Accum)n;
  long _Accum qw = nw / dw;
  int q_camada = (int)qw;

  __ESBMC_assert(q_libc == q_camada,
                 "divilr: divifx agrees with camada's division");
  return 0;
}
