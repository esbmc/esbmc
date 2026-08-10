// Cross-implementation check: LLVM libc's fixed_point::divi<fract> against
// avr-libc's published expectations for rdivi (tests/simulate/stdfix/rdivi-1.c).
//
// Two independent projects implementing the same TR 18037 contract by
// different algorithms (LLVM: Newton-Raphson; avr-libc: AVR assembly).
// Any disagreement is a bug in one of them.
#include "src/__support/fixed_point/fx_bits.h"
#include <limits.h>

extern "C" void __ESBMC_assert(bool, const char *);

using LIBC_NAMESPACE::fixed_point::divi;
using Rep = LIBC_NAMESPACE::fixed_point::FXRep<_Fract>;

int main()
{
  // exact simple fractions
  __ESBMC_assert(divi<_Fract>(0, 1) == 0.0r, "avr: rdivi(0,1) == 0");
  __ESBMC_assert(divi<_Fract>(1, 2) == 0.5r, "avr: rdivi(1,2) == 0.5");
  __ESBMC_assert(divi<_Fract>(1, 4) == 0.25r, "avr: rdivi(1,4) == 0.25");
  __ESBMC_assert(divi<_Fract>(-1, 2) == -0.5r, "avr: rdivi(-1,2) == -0.5");
  __ESBMC_assert(divi<_Fract>(-1, 4) == -0.25r, "avr: rdivi(-1,4) == -0.25");
  __ESBMC_assert(divi<_Fract>(1, -2) == -0.5r, "avr: rdivi(1,-2) == -0.5");
  __ESBMC_assert(divi<_Fract>(1, -4) == -0.25r, "avr: rdivi(1,-4) == -0.25");
  __ESBMC_assert(divi<_Fract>(-1, -2) == 0.5r, "avr: rdivi(-1,-2) == 0.5");
  __ESBMC_assert(divi<_Fract>(-1, -4) == 0.25r, "avr: rdivi(-1,-4) == 0.25");

  // saturation corners
  __ESBMC_assert(divi<_Fract>(1, 1) == Rep::MAX(), "avr: rdivi(1,1) saturates");
  __ESBMC_assert(
    divi<_Fract>(-1, 1) == Rep::MIN(), "avr: rdivi(-1,1) saturates");
  __ESBMC_assert(
    divi<_Fract>(1, -1) == Rep::MIN(), "avr: rdivi(1,-1) saturates");
  __ESBMC_assert(
    divi<_Fract>(-1, -1) == Rep::MAX(), "avr: rdivi(-1,-1) saturates");
  __ESBMC_assert(
    divi<_Fract>(INT_MAX, 1) == Rep::MAX(), "avr: rdivi(INT_MAX,1)");
  __ESBMC_assert(
    divi<_Fract>(INT_MIN, 1) == Rep::MIN(), "avr: rdivi(INT_MIN,1)");
  __ESBMC_assert(
    divi<_Fract>(INT_MAX, INT_MAX) == Rep::MAX(), "avr: rdivi(MAX,MAX)");
  __ESBMC_assert(
    divi<_Fract>(INT_MIN, INT_MIN) == Rep::MAX(), "avr: rdivi(MIN,MIN)");
  __ESBMC_assert(
    divi<_Fract>(INT_MIN, INT_MAX) == Rep::MIN(), "avr: rdivi(MIN,MAX)");
  __ESBMC_assert(
    divi<_Fract>(INT_MAX, INT_MIN) == Rep::MIN(), "avr: rdivi(MAX,MIN)");
  __ESBMC_assert(
    divi<_Fract>(INT_MAX - 1, INT_MAX) == Rep::MAX() - Rep::EPS(),
    "avr: rdivi(MAX-1,MAX) is one ulp below MAX");

  // near epsilon
  __ESBMC_assert(
    divi<_Fract>(1, INT_MAX) == Rep::EPS(), "avr: rdivi(1,INT_MAX) == EPS");
  return 0;
}
