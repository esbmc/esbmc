// divilk against camada's mkFXPDiv, at the widest format libc has.
//
// divilk calls divifx<int, long _Accum>(n, d). long _Accum is s32.31 --
// already the widest fixed-point type, so unlike the _Fract entry points there
// is no wider format to divide in. That is fine here: 31 fraction bits leave
// 32 integer bits, so a small n and a bounded quotient both fit the format
// itself.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_n();
extern "C" long long nondet_ll();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  int n = nondet_n();
  long long db = nondet_ll();
  __ESBMC_assume(db != 0);
  __ESBMC_assume(db > 0);
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db >= (((long long)1) << 27));

  long _Accum d;
  __ESBMC_bitcast(&d, &db);

  int q_libc = fx::divifx<int, long _Accum>(n, d);

  long _Accum nw = (long _Accum)n;
  long _Accum qw = nw / d;
  int q_camada = (int)qw;

  __ESBMC_assert(q_libc == q_camada,
                 "divilk: divifx agrees with camada's division");
  return 0;
}
