#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double big = 1e10;
  assert(asinh(-big) == -INFINITY);
  return 0;
}
