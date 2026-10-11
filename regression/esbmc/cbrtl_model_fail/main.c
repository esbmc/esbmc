#include <assert.h>
#include <math.h>

int main()
{
  long double r = cbrtl(2.0L);
  assert(r > 1.2599L && r < 1.26L);
  assert(r == 1.25L);
  return 0;
}
