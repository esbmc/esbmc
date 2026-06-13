// Smoke test for issue #4438.
//
// Mirrors the shape of c/neural-networks/log_6_safe.c-amalgamation
// (SV-COMP 2026): a nondet float is range-guarded, then passed
// through a libm call.  Under --k-induction --interval-analysis
// --floatbv the interval-derived float bound used to be tagged
// inductive-step-only, so the k-induction base case ignored it and
// could observe spurious NaN values propagated by libm operational
// models.  Post-fix the bound is enforced in every k-induction
// phase.
//
// This test exercises the configuration plumbing only: sqrtf is
// modeled precisely enough that the bug does not fire even pre-fix.
// The actual false-alarm reproducer is the SV-COMP benchmark
// itself, whose solve time (Z3 and Bitwuzla both >> 8 min on the
// fix-validation runs we attempted) makes it unsuitable as a
// regression test.
#include <math.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  if (!(__builtin_isgreaterequal(x, 1.0f) &&
        __builtin_islessequal(x, 10.0f)))
    return 0;

  float y = sqrtf(x);
  if (!(__builtin_isgreaterequal(y, 0.0f) &&
        __builtin_islessequal(y, 10.0f)))
  {
  ERROR:;
  }
  return 0;
}
