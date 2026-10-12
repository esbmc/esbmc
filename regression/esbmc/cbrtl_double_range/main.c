#include <assert.h>
#include <math.h>

int main()
{
  assert(cbrtl(0x1p1020L) == 0x1p340L);
  assert(cbrtl(-0x1p1020L) == -0x1p340L);
  assert(cbrtl(0x1p-1020L) == 0x1p-340L);
  return 0;
}
