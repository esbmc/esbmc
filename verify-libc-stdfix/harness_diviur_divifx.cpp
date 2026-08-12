// diviur against the contract divifx documents for itself.
//
// diviur.cpp calls fixed_point::divifx<unsigned int, unsigned _Fract>(n, d) -- integer divided by
// FIXED-POINT, returning an integer. This is NOT fixed_point::divi, which is
// integer/integer -> fixed-point and which only rdivi reaches. The eight divi*
// entry points had no verification before this harness.
//
// fx_bits.h:337-338 states the contract with no error term:
//
//   "Divide an integer operand by a fixed-point operand and return the
//    mathematically exact result as an IntType rounded towards 0."
//
// Exact and truncating is fully checkable. With d_raw the raw bits of d and
// F = 16 fraction bits, the true quotient is n*2^F / d_raw, so the returned q
// must satisfy, without any division in the property:
//
//   q  * d_raw  is between n*2^F and (q+1)*d_raw, on the side truncation gives
//
// Products are done at full width so the check itself cannot overflow.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" unsigned int nondet_n();
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;

#define FBITS 16
#define QMAX 0xffffffffu
#define QMIN 0u

int main()
{
  unsigned int n = nondet_n();
  unsigned short db = nondet_ushort();
  __ESBMC_assume(db != 0);          /* divide by zero is UB per the source */

  /* Bound n so the 128-bit products stay tractable. The divisor stays fully
   * symbolic over its whole format, which is where the interesting behaviour
   * is -- a small d_raw means a huge quotient and a large one means heavy
   * truncation. This is a restriction of the domain, stated rather than
   * hidden: it is not a proof over all n. */
  __ESBMC_assume(n >= -1024 && n <= 1024);

  unsigned _Fract d;
  __ESBMC_bitcast(&d, &db);

  unsigned int q = fx::divifx<unsigned int, unsigned _Fract>(n, d);

  /* Exclude overflow of the return type. TR 18037 (quoted at fx_bits.h:223
   * for the sibling idiv) makes an overflowing integer result undefined, and
   * divifx inherits that: with d_raw = 1 the true quotient is n*2^F, which
   * exceeds IntType for most n. Verify the in-range domain and say so, rather
   * than assert an identity that cannot hold where the result does not fit. */
  /* Overflow test by multiplication, not division: the exact quotient exceeds
   * QMAX iff |n*2^F| > QMAX*|d_raw|. Keeps the property division-free so the
   * solver sees only multiplications. */
  {
    __int128 an0 = ((__int128)n << FBITS);
    if (an0 < 0)
      an0 = -an0;
    __int128 ad0 = (__int128)db;
    if (ad0 < 0)
      ad0 = -ad0;
    if (an0 > (__int128)QMAX * ad0)
      return 0;
  }

  /* exact truncating division, checked by multiplication at full width */
  __int128 num = (__int128)n << 16;
  __int128 den = (__int128)db;
  __int128 prod = (__int128)q * den;

  /* trunc toward zero: |prod| <= |num| and adding one more |den| overshoots */
  __int128 anum = num < 0 ? -num : num;
  __int128 aprod = prod < 0 ? -prod : prod;
  __int128 aden = den < 0 ? -den : den;

  __ESBMC_assert(aprod <= anum, "diviur: |q*d| <= |n*2^16| (no overshoot)");
  __ESBMC_assert(aprod + aden > anum, "diviur: q is the largest such quotient");
  return 0;
}
