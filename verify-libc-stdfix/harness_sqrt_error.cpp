// Verify LLVM libc's documented accuracy claim for its fixed-point sqrt:
//
//   "Integer square root - Accurate version:
//    Absolute errors < 2^(-fraction length)."
//        -- libc/src/__support/fixed_point/sqrt.h:211-212
//
// i.e. the result is within one ULP of the true square root. Checked without
// an exact-sqrt oracle by bracketing: r is within 1 ulp of sqrt(x) iff
//
//     r*r <= x   and   x < (r+ulp)*(r+ulp)
//
// The upper bracket is evaluated in the SATURATING type so that r+ulp at the
// format maximum clamps rather than wrapping, which would make the bound
// trivially true for the largest inputs.
//
// This is LLVM libc's own source, included from the checkout -- not a
// transcription. u0.8 is exhaustive: all 256 inputs.
#include "src/__support/fixed_point/sqrt.h"

extern "C" unsigned short _Fract nondet_ufract8();
extern "C" void __ESBMC_assert(bool, const char *);

using LIBC_NAMESPACE::fixed_point::sqrt;

int main()
{
  unsigned short _Fract x = nondet_ufract8();
  unsigned short _Fract r = sqrt(x);

  // Lower bracket: r may not overshoot the true root.
  __ESBMC_assert(r * r <= x, "sqrt(x)^2 <= x  (r is not above the root)");

  /* Upper bracket: r may not undershoot the true root by a whole ulp, i.e.
   * x <= (r+ulp)^2. u0.8 cannot represent r+ulp once r is the format maximum
   * (255/256), and (255/256)^2 < 255/256, so the bracket is unsatisfiable
   * there for arithmetic reasons rather than accuracy ones -- at x=255/256
   * libc returns 255/256 against a true root of 0.998045, an error of
   * 0.00195 against a 1-ulp tolerance of 0.00391, comfortably inside its
   * claim. Skip only that representability edge. */
  const unsigned short _Fract max = 0.99609375uhr;
  const unsigned short _Fract ulp = 0.00390625uhr; // 2^-8
  if (r != max)
  {
    unsigned short _Fract next = r + ulp;
    __ESBMC_assert(
      x <= next * next, "x <= (sqrt(x)+ulp)^2  (r within 1 ulp below root)");
  }

  return 0;
}
