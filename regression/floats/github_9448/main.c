/* github #9448: an in-range conversion stays exact next to an undefined one
 * under --fp2bv. */
#include <assert.h>
#include <math.h>

float nondet_float(void);

int main(void)
{
  float f1 = nondet_float(), f2 = nondet_float();
  __ESBMC_assume(isnan(f1) && f2 >= 1.0f && f2 < 2.0f);
  int a = (int)f1;
  int b = (int)f2;
  assert(b == 1);
  return 0;
}
