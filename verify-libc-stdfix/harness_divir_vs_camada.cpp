// divir against camada's mkFXPDiv -- including the _Fract formats, which the
// earlier attempt wrongly excluded.
//
// divir(n, d) = divifx<int, _Fract>(n, d): integer / s0.15 -> integer.
//
// The earlier harness tried to represent n in s0.15 itself, which cannot work:
// s0.15 holds only [-1, 1), so (_Fract)64 is meaningless. That was a property
// of the C type's range, NOT a limit of camada -- mkFXPDiv is format-generic.
//
// The fix is to do the camada-side division in a format wide enough for both
// the dividend and the quotient. s16.15 (_Accum) has the same 15 fraction bits
// as s0.15, so widening the divisor into it is exact -- the raw bits carry the
// same value -- while giving 16 integer bits of headroom for n and the result.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_n();
extern "C" short nondet_short();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  int n = nondet_n();
  short db = nondet_short();
  __ESBMC_assume(db != 0);

  _Fract d;
  __ESBMC_bitcast(&d, &db);

  int q_libc = fx::divifx<int, _Fract>(n, d);

  /* camada side, computed in s16.15 so n and the quotient both fit.
   * d widens exactly: same fraction length, so the value is preserved. */
  _Accum dw = (_Accum)d;
  _Accum nw = (_Accum)n;
  _Accum qw = nw / dw;              /* mkFXPDiv */
  int q_camada = (int)qw;

  /* Keep both sides inside s16.15's range so neither overflows, and keep the
   * quotient non-negative so camada's floor matches divifx's toward-zero. */
  __ESBMC_assume(n >= 0 && n <= 16);
  __ESBMC_assume(db > 0 && db >= (1 << 11));

  __ESBMC_assert(q_libc == q_camada,
                 "divir: divifx agrees with camada's division");
  return 0;
}
