// LLVM libc's exphk against camada's mkFXPExp, inside the solver.
//
// mkFXPExp is exp correctly rounded to nearest (ties to even), saturating at
// the format MAX and flushing to zero below half an ulp. Validated before use
// against eight natively-measured anchors spanning the range, including the
// saturation and flush-to-zero boundaries -- see RESULTS.md.
//
// exphk's documented claim is a RELATIVE bound, and it is stated for one step
// of the range reduction rather than end to end:
//
//     exp(x) ~ exp(hi)*exp(mid)*(1 + lo)   "with relative errors < |lo|^2 <= 2^-8"
//
// So the honest property is not "libc equals the oracle". It is: the relative
// difference from the exact value stays under 2^-8. Since both sides are in
// s8.7 the comparison is done on raw integers scaled up, so no fixed-point
// rounding enters the bound itself:
//
//     |libc - exact| * 256 <= exact          (i.e. rel err <= 2^-8)
//
// The oracle is the correctly-rounded exact value, so it stands in for `exact`
// with at most half an ulp of its own -- accounted for by the +1 slack below.
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/CPP/bit.h"
#include "hdr/stdint_proxy.h"

extern "C" short _Accum __ESBMC_fxp_exp_hk(short _Accum);
extern "C" short nondet_short();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;
namespace cpp = LIBC_NAMESPACE::cpp;

// libc's exphk body, from src/stdfix/exphk.cpp
static constexpr short accum EXP_HI[12] = {
  0x1.0p-7hk, 0x1.0p-6hk, 0x1.8p-5hk,  0x1.1p-3hk,  0x1.78p-2hk,  0x1.0p0hk,
  0x1.5cp1hk, 0x1.d9p2hk, 0x1.416p4hk, 0x1.b4dp5hk, 0x1.28d4p7hk, SACCUM_MAX,
};
static constexpr short accum EXP_MID[8] = {
  0x1.38p-1hk, 0x1.6p-1hk, 0x1.9p-1hk, 0x1.c4p-1hk,
  0x1.0p0hk,   0x1.22p0hk, 0x1.48p0hk, 0x1.74p0hk,
};
static short accum exphk_body(short accum x)
{
  using FXRep = fx::FXRep<short accum>;
  using StorageType = typename FXRep::StorageType;
  if (x >= 0x1.64p2hk)
    return FXRep::MAX();
  if (x <= -0x1.63p2hk)
    return FXRep::ZERO();
  constexpr short accum ONE_SIXTEENTH = 0x1.0p-4hk;
  short accum x_rounded =
    ((x + ONE_SIXTEENTH) >> (FXRep::FRACTION_LEN - 3))
    << (FXRep::FRACTION_LEN - 3);
  short accum lo = x - x_rounded;
  StorageType indices = cpp::bit_cast<StorageType>(
    (x_rounded + 0x1.6p2hk) >> (FXRep::FRACTION_LEN - 3));
  short accum exp_hi = EXP_HI[indices >> 3];
  short accum exp_mid = EXP_MID[indices & 0x7];
  return (exp_hi * (exp_mid * (0x1.0p0hk + lo)));
}

int main()
{
  short xb = nondet_short();
  short _Accum x;
  __ESBMC_bitcast(&x, &xb);

  short _Accum lr = exphk_body(x);
  short _Accum rr = __ESBMC_fxp_exp_hk(x);

  short lb, rb;
  __ESBMC_bitcast(&lb, &lr);
  __ESBMC_bitcast(&rb, &rr);

  // Stay strictly inside the representable band: where the exact value
  // saturates or flushes, "relative error" is not the claim being made.
  __ESBMC_assume(rb > 0 && rb < 32767);

  int diff = (int)lb - (int)rb;
  if (diff < 0)
    diff = -diff;

  // relative error <= 2^-8, with +1 for the oracle's own half ulp
  __ESBMC_assert(
    diff * 256 <= (int)rb + 1,
    "exphk: relative error within the documented 2^-8");
  return 0;
}
