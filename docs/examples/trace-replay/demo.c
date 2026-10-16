#include <assert.h>

int main(void)
{
  int x = 1;
  int y = x - 1;
  x = y + 2;
  assert(x == 3);
  return 0;
}
