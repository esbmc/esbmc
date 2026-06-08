#include <assert.h>
#include <complex.h>

int main()
{
  /* |3 + 4i| = sqrt(9 + 16) = 5 */
  double complex z = 3.0 + 4.0 * I;

  assert(cabs(z) == 5.0);

  return 0;
}
