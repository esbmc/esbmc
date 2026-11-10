// #10172: the truncated quotient 0x1p-53 is not the correctly rounded one.
#include <assert.h>
int main()
{
  double c = 1.0 / 9007199254740991.0;
  assert(c == 0x1p-53);
  return 0;
}
