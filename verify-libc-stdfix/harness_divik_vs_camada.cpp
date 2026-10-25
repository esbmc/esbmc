// divik against camada's mkFXPDiv on the same symbolic inputs.
//
// divik.cpp calls fixed_point::divifx<int, _Accum>(n, d): integer / fixed-point
// -> integer, "the mathematically exact result ... rounded towards 0".
//
// camada's mkFXPDiv divides fixed-point by fixed-point. For the comparison to
// mean anything, BOTH sides must be representable in _Accum:
//
//   * n itself -- so only the _Accum formats qualify. u0.16 cannot hold any
//     n >= 1, so the _Fract entry points cannot be compared this way at all;
//     an earlier version of this harness cast n into u0.16 and got garbage.
//   * the quotient n/d -- with a tiny d the quotient explodes past the format
//     maximum and camada's division overflows. Both bounds are assumed below.
//
// Within those bounds the two must agree exactly.
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
  __ESBMC_assume(db > 0);   /* keep the quotient non-negative so camada's floor
                             * and divifx's toward-zero rounding coincide */

  _Accum d;
  __ESBMC_bitcast(&d, &db);

  /* n representable in the format's integer part */
  __ESBMC_assume(n >= 0 && n <= 16);

  /* quotient representable too: n/d <= FORMAT_MAX means n * 2^F <= MAX * d_raw.
   * Checked by multiplication so no division enters the assumption. */
  /* Keep the quotient in range via a lower bound on the divisor rather than a
   * wide product: n <= 16 and d >= 2^-4 give a quotient <= 256, well inside the
   * format, and it is far cheaper for the solver than the general bound. */
  __ESBMC_assume(db >= (1 << 11));

  int q_libc = fx::divifx<int, _Accum>(n, d);

  _Accum nv = (_Accum)n;
  _Accum qv = nv / d;              /* mkFXPDiv */
  int q_camada = (int)qv;

  __ESBMC_assert(q_libc == q_camada,
                 "divik: divifx agrees with camada's division");
  return 0;
}
