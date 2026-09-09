// CONTROL for harness_divi_bug1.cpp, and it must VERIFY SUCCESSFUL.
//
// Same signs and same magnitudes as the bug-1 harness, but the divisor is
// NOT a power of two, so divi takes its general branch. That branch derives
// the sign from both operands and is correct.
//
// This is what makes bug 1 a real defect rather than a wrong property: if
// this harness also failed, the sign law as stated -- or the harness -- would
// be the thing at fault, not the power-of-two branch.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---- the general (non-power-of-two) branch, as a CONTROL: same signs,
   * same magnitudes, but a divisor that is not a power of two. If this also
   * failed, the defect would not be branch-specific. ---- */
  {
    int n = nondet_int(), d = nondet_int();
    __ESBMC_assume(d == -3 || d == -5 || d == -6 || d == -7);
    __ESBMC_assume(n < 0 && n > -128);
    __ESBMC_assume(-n >= -d);

    short _Fract r = fx::divi<short _Fract>(n, d);
    int8_t raw = bit_cast<int8_t, short _Fract>(r);
    __ESBMC_assert(raw > 0, "divihr control: general branch sign is correct");
  }

  return 0;
}
