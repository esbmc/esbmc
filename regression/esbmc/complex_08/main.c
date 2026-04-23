#include <assert.h>
#include <complex.h>

int main()
{
  /* arg(1 + 0i) = atan2(0, 1) = 0 */
  double complex z = 1.0 + 0.0 * I;

  assert(carg(z) == 0.0);

  return 0;
}
