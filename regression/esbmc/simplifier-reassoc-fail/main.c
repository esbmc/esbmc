#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // Same shape as the passing test, but with a wrong expected value.
  int z = 2 - x - 10 * 2 - y * 4;
  assert(z == -17 - x - 4 * y);

  return 0;
}
