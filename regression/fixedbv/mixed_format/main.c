/* Mixed-format operands: TR 18037 computes in the common format without
 * double rounding. Values pinned by native execution. */
#include <assert.h>

int main(void)
{
  short _Fract mf = -0.5hr;
  _Accum mk2 = 1.25k;
  assert(mf + mk2 == 0.75k);
  assert(mk2 * mf == -0.625k);
  return 0;
}
