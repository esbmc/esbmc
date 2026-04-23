#include <assert.h>
#include <complex.h>

int main()
{
  /* |3 + 4i| = 5, not 99 */
  double complex z = 3.0 + 4.0 * I;

  assert(cabs(z) == 99.0);

  return 0;
}
