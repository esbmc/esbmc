// diviulk against camada's mkFXPDiv, at the widest format libc has.
//
// diviulk calls divifx<unsigned int, unsigned long _Accum>(n, d). unsigned long _Accum is u32.32 --
// already the widest fixed-point type, so unlike the _Fract entry points there
// is no wider format to divide in. That is fine here: 32 fraction bits leave
// 32 integer bits, so a small n and a bounded quotient both fit the format
// itself.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_n();
extern "C" unsigned long long nondet_ull();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  unsigned int n = nondet_n();
  unsigned long long db = nondet_ull();
  __ESBMC_assume(db != 0);
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db >= (((unsigned long long)1) << 28));

  unsigned long _Accum d;
  __ESBMC_bitcast(&d, &db);

  unsigned int q_libc = fx::divifx<unsigned int, unsigned long _Accum>(n, d);

  unsigned long _Accum nw = (unsigned long _Accum)n;
  unsigned long _Accum qw = nw / d;
  unsigned int q_camada = (unsigned int)qw;

  __ESBMC_assert(q_libc == q_camada,
                 "diviulk: divifx agrees with camada's division");
  return 0;
}
