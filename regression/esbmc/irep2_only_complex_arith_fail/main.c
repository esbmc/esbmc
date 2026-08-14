#include <assert.h>
#include <complex.h>

int main(void)
{
  /* (4+2i) / (1+1i) = 3 - 1i, not 99: a sign error in the imaginary
     component lands here rather than on the shape of the lowering. */
  double complex z = (4.0 + 2.0 * I) / (1.0 + 1.0 * I);
  assert(cimag(z) == 99.0);

  return 0;
}
