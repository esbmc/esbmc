#include <assert.h>
#include <math.h>

/* ln(1.125) = 0.11778303565638345574... The truncated Taylor series in
   src/c2goto/library/libm/exp.c reaches this accuracy at its default 8 terms
   and not at 6, so this program and its --fp-taylor-terms 6 sibling separate
   the two settings. */
int main()
{
  double y = log1p(0.125);
  assert(fabs(y - 0.11778303565638346) < 1e-12);
  return 0;
}
