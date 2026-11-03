#include <assert.h>
#include <math.h>

int main()
{
  double r = cbrt(2.0);
  assert(r > 1.2599 && r < 1.26);
  assert(r == 1.25);
  return 0;
}
