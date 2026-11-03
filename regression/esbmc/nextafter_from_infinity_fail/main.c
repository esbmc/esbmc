#include <assert.h>
#include <float.h>
#include <math.h>

int main()
{
  double r = nextafter(INFINITY, 0.0);
  assert(r == DBL_MAX);
  assert(isinf(r));
  return 0;
}
