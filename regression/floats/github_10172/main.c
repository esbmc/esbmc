// #10172: 1 / (2^53 - 1) lies just above a halfway point and must round up.
#include <assert.h>
int main()
{
  double c = 1.0 / 9007199254740991.0;
  assert(c == 0x1.0000000000001p-53);
  return 0;
}
