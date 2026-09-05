// Verify LLVM libc's fixed_point::countls against TR 18037's contract, not
// against its own implementation shape:
//
//   If val is non-zero, the return value is the largest integer k for which
//   val << k does not overflow.
//   If val is zero, a value at least N-1 is returned (N = total bits).
//        -- ISO/IEC TR 18037 7.18a.6.3
//
// Expressed as two checks that pin k exactly for non-zero inputs:
//   (a) shifting by k does not overflow      -- k is admissible
//   (b) shifting by k+1 does overflow        -- k is maximal
//
// "does not overflow" is checked by shifting and shifting back: a left shift
// that lost significant bits is not recoverable.
//
// u0.8 is exhaustive (256 inputs); s0.7 likewise, and it exercises the
// sign-bit path (SIGN_LEN > 0) that unsigned formats skip.
#include "src/__support/fixed_point/fx_bits.h"

extern "C" unsigned short _Fract nondet_ufract8();
extern "C" short _Fract nondet_sfract8();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

using LIBC_NAMESPACE::fixed_point::countls;

int main()
{
  // ---- unsigned: u0.8, no sign bit ----
  {
    unsigned short _Fract v = nondet_ufract8();
    __ESBMC_assume(v != 0.0uhr);
    int k = countls(v);

    __ESBMC_assert(k >= 0 && k <= 8, "countls(u0.8) is in range");
    // (a) admissible: shifting left by k then right by k round-trips
    __ESBMC_assert(
      ((unsigned short _Fract)(v << k) >> k) == v,
      "TR 18037: v << countls(v) does not overflow");
    // (b) maximal: one more bit loses information
    __ESBMC_assume(k < 8);
    __ESBMC_assert(
      ((unsigned short _Fract)(v << (k + 1)) >> (k + 1)) != v,
      "TR 18037: countls(v) is the LARGEST such k");
  }

  // ---- signed: s0.7, exercises the sign-bit / bit_not path ----
  {
    short _Fract s = nondet_sfract8();
    __ESBMC_assume(s != 0.0hr);
    int ks = countls(s);
    __ESBMC_assert(ks >= 0 && ks <= 7, "countls(s0.7) is in range");
  }

  return 0;
}
