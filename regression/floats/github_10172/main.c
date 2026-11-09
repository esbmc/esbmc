#include <assert.h>

int main()
{
  double above_tie = 1.0 / 9007199254740991.0;
  assert(above_tie == 0x1.0000000000001p-53);

  double subnormal_dividend = 0x0.0000000000003p-1022 / 0x1.56e1fc2f8f359p-997;
  assert(subnormal_dividend == 0x1.1eb2d66005835p-76);
  return 0;
}
