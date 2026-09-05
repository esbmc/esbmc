// BUG 1 of 3 in LLVM libc's fixed_point::divi -- the SIGN LAW, proved over
// all inputs at s0.7.
//
// fx_bits.h:257  res64 = (n << F) >> k     <- already carries n's sign
// fx_bits.h:268  res = (d<0) ? -1*res : res <- negates AGAIN on d's sign
//
// For n < 0 and d < 0 the sign is applied twice, so a positive quotient is
// returned negative. Only the power-of-two-divisor branch is affected; the
// general branch derives `result_is_negative` from both signs correctly.
// harness_divi_control.cpp is the CONTROL that pins the defect to this
// branch, and it VERIFIES SUCCESSFUL.
//
// Only the sign is asserted. divi's magnitude has no documented accuracy
// bound (Newton-Raphson with per-iteration error comments only), so a
// few-ulp magnitude error is not claimed as a defect anywhere here.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---- BUG 1: s0.7, power-of-two divisor, both operands negative ---- */
  {
    int n = nondet_int(), d = nondet_int();
    // A negative power-of-two divisor, and a numerator that makes the exact
    // quotient at least +1.0 so the correct answer is the MAX rail.
    __ESBMC_assume(d == -1 || d == -2 || d == -4 || d == -8 || d == -16);
    __ESBMC_assume(n < 0 && n > -128);
    __ESBMC_assume(-n >= -d); // |n| >= |d|, so the exact quotient is >= 1.0

    short _Fract r = fx::divi<short _Fract>(n, d);
    int8_t raw = bit_cast<int8_t, short _Fract>(r);

    // n<0, d<0 -> the quotient is POSITIVE. This is the sign law, and it
    // needs no accuracy contract to state.
    __ESBMC_assert(raw > 0, "divihr: neg/neg is positive");
  }

  return 0;
}
