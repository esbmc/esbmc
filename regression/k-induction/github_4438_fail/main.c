// Negative companion to github_4438: same shape but the downstream
// gate is genuinely violable for the bounded input range, so the
// k-induction base case must still discover the ERROR label even
// after the issue #4438 fix forced the interval-derived float
// assume into the base case.  Confirms the fix does not over-
// constrain the base case into masking real errors.
#include <math.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  if (!(__builtin_isgreaterequal(x, 1.0f) &&
        __builtin_islessequal(x, 100.0f)))
    return 0;

  float y = sqrtf(x);
  if (!(__builtin_isgreaterequal(y, 0.0f) &&
        __builtin_islessequal(y, 5.0f)))
  {
  ERROR:;
  }
  return 0;
}
