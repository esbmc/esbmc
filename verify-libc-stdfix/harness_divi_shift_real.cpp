// The 1 << F undefined behaviour, caught by ESBMC's own UB checker on libc's
// REAL divi template -- not on a reduced hand-written shift, and with no
// assertion of my own.
//
// The earlier harness (harness_divi_shift.c) asserted `scale > 0`, which is a
// hand-written CONSEQUENCE of the UB rather than the UB itself. That was weaker
// than necessary: --overflow-check flags the shift directly, reporting
// "arithmetic overflow on shl" with CWE-190/191.
//
// fx_bits.h:266-267, reached whenever divi is instantiated at a format with
// F >= 31 and given a power-of-two divisor:
//
//     long accum res_accum = static_cast<long accum>(res64)
//                          / static_cast<long accum>(1 << F);
//
// `1` is an int, so F == 31 overflows into the sign bit (C11 6.5.7p4) and
// F == 32 shifts by the full width (6.5.7p3).
//
// Run with:  esbmc harness_divi_shift_real.cpp --overflow-check ...
// No __ESBMC_assert appears below on purpose -- the checker is the oracle.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  int n = nondet_int();
  int d = nondet_int();

  /* a power-of-two divisor takes the fast path containing the shift */
  __ESBMC_assume(d == 2 || d == 4 || d == -2 || d == -4);
  __ESBMC_assume(n >= -8 && n <= 8);

  /* long _Fract has F == 31: the shift is `1 << 31`. */
  volatile long _Fract r = fx::divi<long _Fract>(n, d);
  (void)r;
  return 0;
}
