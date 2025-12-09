#include <assert.h>

int main()
{
  assert(ffs(0) == 0);
  assert(ffs(1) == 1);
  assert(ffs(2) == 2);
  assert(ffs(4) == 3);
  assert(ffs(8) == 4);

  assert(ffs(3) == 1);
  assert(ffs(6) == 2);
  assert(ffs(12) == 3);

  assert(ffs(0x1000) == 13);
  assert(ffs(0x80000000) == 32);

  assert(ffs(0b1011000) == 4);
  assert(ffs(0b01010000) == 5);

  return 0;
}
