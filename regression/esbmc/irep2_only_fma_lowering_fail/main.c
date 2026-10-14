#include <assert.h>
#include <math.h>

/* C17 7.12.13.1: fma rounds once, so the product's tail survives the addition
 * and the fused result differs from the separately rounded one. */
int main(void)
{
  double a = 0x1.fffffffffffffp0;
  double c = -0x1.fffffffffffffp1;
  assert(fma(a, a, c) == a * a + c);
  return 0;
}
