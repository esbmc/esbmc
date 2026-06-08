#include <assert.h>
#include <complex.h>

int main()
{
  /* (1+2i) * (3+4i) = -5 + 10i, not 99 */
  double complex a = 1.0 + 2.0 * I;
  double complex b = 3.0 + 4.0 * I;
  double complex z = a * b;

  assert(creal(z) == 99.0);

  return 0;
}
