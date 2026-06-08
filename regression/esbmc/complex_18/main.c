#include <assert.h>
#include <complex.h>

int main()
{
  /* (4+2i) / (1+1i) = 3 - 1i, not 99 */
  double complex a = 4.0 + 2.0 * I;
  double complex b = 1.0 + 1.0 * I;
  double complex z = a / b;

  assert(cimag(z) == 99.0);

  return 0;
}
