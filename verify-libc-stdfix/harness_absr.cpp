// Verify LLVM libc's fixed_point::abs<fract> -- the REAL source, included
// from the llvm-project checkout, not a transcription.
//
// Properties (TR 18037 7.18a.6.1 absfx):
//   P1  result is never negative
//   P2  result equals |x| except at MIN, where it saturates to MAX
//   P3  idempotent
#include "src/__support/fixed_point/fx_bits.h"

extern "C" short _Fract nondet_sfract();
extern "C" void __ESBMC_assert(bool, const char *);

using LIBC_NAMESPACE::fixed_point::FXRep;

int main()
{
  short _Fract x = nondet_sfract();
  short _Fract r = LIBC_NAMESPACE::fixed_point::abs(x);

  using Rep = FXRep<short _Fract>;

  __ESBMC_assert(r >= Rep::ZERO(), "P1: abs is never negative");
  __ESBMC_assert(
    x == Rep::MIN() ? r == Rep::MAX() : (r == x || r == -x),
    "P2: magnitude preserved, MIN saturates");
  __ESBMC_assert(
    LIBC_NAMESPACE::fixed_point::abs(r) == r, "P3: abs is idempotent");
  return 0;
}
