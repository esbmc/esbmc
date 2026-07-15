#include <assert.h>
#include <math.h>
int main()
{
  double z = 0.0;
  assert(fpclassify(z) == FP_NORMAL); // wrong: 0.0 is FP_ZERO, not FP_NORMAL
  return 0;
}
