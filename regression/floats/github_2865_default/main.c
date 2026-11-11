#include <assert.h>
#include <math.h>

/* e^0.24 - 1 = 0.27124915032140468... The truncated Taylor series in
   src/c2goto/library/libm/exp.c reaches this accuracy at its default 8 terms
   and not at 6, so this program and its --fp-taylor-terms 6 sibling separate
   the two settings. */
int main()
{
  double y = expm1(0.24);
  assert(fabs(y - 0.27124915032140468) < 1e-12);
  return 0;
}
