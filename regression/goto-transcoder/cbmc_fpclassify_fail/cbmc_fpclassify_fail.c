// Negative companion to cbmc_fpclassify: the linked __fpclassifyd body must
// return FP_ZERO for 0.0, so the (wrong) FP_NORMAL expectation below must be
// violated. See cbmc_fpclassify.c for why the internal symbol is called
// directly.
#include <assert.h>
#include <math.h>
extern int __fpclassifyd(double);
int main()
{
  double z = 0.0;
  assert(__fpclassifyd(z) == FP_NORMAL); // wrong: 0.0 is FP_ZERO, not FP_NORMAL
  return 0;
}
