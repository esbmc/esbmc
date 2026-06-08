#include <assert.h>
#include <complex.h>

int main()
{
  /* arg(1 + 0i) = 0, not 99 */
  double complex z = 1.0 + 0.0 * I;

  assert(carg(z) == 99.0);

  return 0;
}
