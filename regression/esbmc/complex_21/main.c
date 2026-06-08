#include <assert.h>
#include <complex.h>

int main()
{
  /* 0+0i converted to _Bool is 0 */
  double complex z = 0.0 + 0.0 * I;

  _Bool b = (_Bool)z;

  assert(b == 0);

  return 0;
}
