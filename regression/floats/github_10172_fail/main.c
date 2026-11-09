#include <assert.h>

int main()
{
  double above_tie = 1.0 / 9007199254740991.0;
  assert(above_tie == 0x1p-53);
  return 0;
}
