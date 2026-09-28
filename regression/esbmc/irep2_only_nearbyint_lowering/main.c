#include <assert.h>
#include <math.h>

int main(void)
{
  assert(nearbyint(2.5) == 2.0);
  assert(nearbyint(3.5) == 4.0);
  assert(nearbyint(-2.5) == -2.0);
  return 0;
}
