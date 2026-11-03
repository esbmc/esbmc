#include <assert.h>
#include <math.h>

int main()
{
  assert(ilogb(8.0) == 3);
  assert(logb(0.75) == 0.0);
  return 0;
}
