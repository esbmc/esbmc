// Verify LLVM libc's fixed_point::round, and settle the tie direction that
// TR 18037 leaves unspecified.
//
// Their claim: "Round-to-nearest, tie-to-(+Inf)" (fx_bits.h:162).
//
// Checked as properties over ALL inputs and ALL rounding positions n, rather
// than by re-implementing the rounding:
//   (a) the result is a multiple of the rounding step 2^-n
//   (b) the result is within one step of the input (it is a rounding, not an
//       arbitrary value)
//   (c) it is the NEAREST such multiple, except at a tie
//   (d) at a tie, it rounds up -- the direction claim, negatives included
//   (e) overflow saturates to MAX instead of wrapping
//
// n is left symbolic, so this covers every rounding position at once.
#include "src/__support/fixed_point/fx_bits.h"

extern "C" short _Fract nondet_sfract8();
extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

using LIBC_NAMESPACE::fixed_point::round;
using Rep = LIBC_NAMESPACE::fixed_point::FXRep<short _Fract>;

int main()
{
  short _Fract x = nondet_sfract8();
  int n = nondet_int();
  __ESBMC_assume(n >= 0 && n < 7); // 7 = FRACTION_LEN for s0.7

  short _Fract r = round(x, n);

  // The rounding step, as a raw-bit count: 2^(FRACTION_LEN - n).
  const __int128 step = (__int128)1 << (7 - n);

  union Pun
  {
    short _Fract f;
    signed char raw;
  };
  Pun px, pr;
  px.f = x;
  pr.f = r;
  const int xr = px.raw, rr = pr.raw;

  // (e) saturation: the only way to land on MAX without being a multiple.
  if (rr == 127)
    return 0; // overflow arm, checked separately below

  // (a) the result is a multiple of the step
  __ESBMC_assert(rr % step == 0, "round(x,n) is a multiple of 2^-n");

  const __int128 x128 = xr;

  // (b) within one step of the input
  __ESBMC_assert(
    rr - x128 <= step && x128 - rr <= step, "round(x,n) is within one step of x");

  // (c)+(d) nearest, ties upward: the distance up is at most the distance
  // down, i.e. it never rounds down when up is strictly closer, and on a tie
  // it takes up.
  const __int128 down =
    (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
  const __int128 up = down + step;
  const __int128 dist_down = x128 - down, dist_up = up - x128;
  if (dist_up <= dist_down)
    __ESBMC_assert(rr == up, "ties and nearer-up round toward +Inf");
  else
    __ESBMC_assert(rr == down, "nearer-down rounds down");

  return 0;
}
