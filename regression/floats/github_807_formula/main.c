/* github #807: a 32-bit integer always fits float's exponent range, so its
 * conversion under --fp2bv needs no denormal shift (bvlshr) in the formula. */
#include <assert.h>

int nondet_int(void);

int main(void)
{
  float f = (float)nondet_int();
  assert(f != 0.5f);
  return 0;
}
