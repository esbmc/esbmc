/*
 * Regression test for GitHub issue #2789.
 * Safe shift operations with constant distances should pass
 * verification even with the default UB shift check enabled.
 */

#include <assert.h>

int main()
{
  int a = 256 << 1;
  assert(a == 512);

  int b = 1024 >> 2;
  assert(b == 256);

  unsigned int c = 1u << 31;
  assert(c == 2147483648u);

  return 0;
}
