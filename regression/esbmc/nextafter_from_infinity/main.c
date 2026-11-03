#include <assert.h>
#include <float.h>
#include <math.h>

int main()
{
  assert(nextafter(INFINITY, 0.0) == DBL_MAX);
  assert(nextafter(-INFINITY, 0.0) == -DBL_MAX);
  assert(nextafterf(INFINITY, 0.0f) == FLT_MAX);
  assert(nextafterf(-INFINITY, 1.0f) == -FLT_MAX);
  assert(nextafter(INFINITY, INFINITY) == INFINITY);
  return 0;
}
