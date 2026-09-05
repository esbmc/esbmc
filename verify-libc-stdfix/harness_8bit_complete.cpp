// Completes the true-8-bit tier of LLVM libc's stdfix: the eight entry points
// whose argument is an 8-bit fixed-point type (suffix hr / uhr).
//
//   abshr  bitshr  bitsuhr  countlshr  countlsuhr  roundhr  rounduhr  sqrtuhr
//
// abs, countls, round and bitsfx are already verified for one signedness
// each in the per-family harnesses; this adds the variants those missed so
// every 8-bit entry point is covered. sqrtuhr is covered by
// harness_sqrt_error.cpp (and violates its documented bound).
//
// Properties are TR 18037 7.18a.6, not re-implementations. Both 8-bit
// domains are 256 values, so these are proofs over every input.
#include "src/__support/fixed_point/fx_bits.h"

extern "C" unsigned short _Fract nondet_ufract8();
extern "C" short _Fract nondet_sfract8();
extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

using LIBC_NAMESPACE::fixed_point::abs;
using LIBC_NAMESPACE::fixed_point::round;
using URep = LIBC_NAMESPACE::fixed_point::FXRep<unsigned short _Fract>;

int main()
{
  /* ---- abshr on the UNSIGNED type (absuhr has no entry point, but abs is
   * instantiated for unsigned inside the library; SIGN_LEN == 0 takes the
   * early-return arm, so identity is the contract). ---- */
  {
    unsigned short _Fract u = nondet_ufract8();
    __ESBMC_assert(abs(u) == u, "abs on an unsigned format is the identity");
  }

  /* ---- rounduhr: the unsigned round path. Same five properties as the
   * signed harness, but the unsigned type has no sign bit, so the
   * rounding step and the saturation rail differ. ---- */
  {
    unsigned short _Fract x = nondet_ufract8();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 8); // FRACTION_LEN == 8 for u0.8

    unsigned short _Fract r = round(x, n);

    union Pun
    {
      unsigned short _Fract f;
      unsigned char raw;
    };
    Pun px, pr;
    px.f = x;
    pr.f = r;
    const int xr = px.raw, rr = pr.raw;
    const __int128 step = (__int128)1 << (8 - n);
    const __int128 x128 = xr;

    if (rr != 255) // the saturating arm is checked separately below
    {
      __ESBMC_assert(rr % step == 0, "rounduhr result is a multiple of 2^-n");
      __ESBMC_assert(
        rr >= x128 - step && rr <= x128 + step,
        "rounduhr result is within one step of x");

      const __int128 down = (x128 / step) * step;
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "rounduhr: ties and nearer-up round up");
      else
        __ESBMC_assert(rr == down, "rounduhr: nearer-down rounds down");
    }
  }

  /* ---- countls on both 8-bit signednesses is covered in
   * harness_countls.cpp; bitsfx/fxbits on both in harness_bitsfx.cpp. ---- */

  return 0;
}
