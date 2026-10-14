/* Round-trip long long -> double -> long long across the double precision
 * boundary (2^53), round-up tie case.
 * At binade k=1 (values in [2^53, 2^54)), representable doubles are spaced by 2.
 * 9007199254740995 = 2^53 + 3 is halfway between 9007199254740994 and
 * 9007199254740996.  Its floor significand index is
 * 9007199254740994/2 = 4503599627370497 (odd), so RNE rounds UP to
 * 9007199254740996.  After the round-trip the result cannot equal
 * 9007199254740995.
 *
 * Regression: under --ir-ieee, symbolic integer-to-double casts must apply
 * target-type precision rounding rather than preserving the integer as an
 * exact SMT real. */
#include <assert.h>

long long nondet_longlong(void);

int main(void)
{
  long long x = nondet_longlong();
  __ESBMC_assume(x >= 9007199254740994LL && x <= 9007199254740996LL);
  double d = (double)x;
  long long y = (long long)d;
  assert(y != 9007199254740995LL);
  return 0;
}
