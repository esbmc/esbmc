// #10154: the second loop is never entered, so its entry assumption in the
// inductive step dropped the path to the failing assertion.
#include <assert.h>

int main(void)
{
  int i, x, y = 0, z = 0, s = 0;
  for (i = 0; i < 1000; i++)
    s = s + 1;
  for (x = 0; x < z; x++)
    y = 1;
  assert(y == 1 || s < 0);
  return 0;
}
