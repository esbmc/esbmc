#include <assert.h>
#include <complex.h>

int main()
{
  /* (1+2i) * (3+4i) = (3-8) + (4+6)i = -5 + 10i */
  double complex a = 1.0 + 2.0 * I;
  double complex b = 3.0 + 4.0 * I;
  double complex z = a * b;

  assert(creal(z) == -5.0);
  assert(cimag(z) == 10.0);

  return 0;
}
