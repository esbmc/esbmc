#include <assert.h>
#include <complex.h>

int main()
{
  double complex z = 3.0 + 2.0 * I;

  assert(creal(z) == 99.0);

  return 0;
}
