#include <assert.h>
#include <math.h>

int main()
{
  long double r = cbrtl(0x1p1020L);
  assert(r == 0x1p340L);
  assert(r == 0x1p341L);
  return 0;
}
