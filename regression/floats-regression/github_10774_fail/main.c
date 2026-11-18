/* #10774: --interval-analysis simplifies the unused libm models, whose
 * remainder() lowers to ieee_rem; under --fixedbv that node is fixedbv. */
#include <math.h>
#include <assert.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  if (x > 3)
    assert(x != 4);
  return 0;
}
