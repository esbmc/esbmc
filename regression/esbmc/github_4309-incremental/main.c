#include <assert.h>

int main(void)
{
  int x;
  if (x > 0)
    x--;
  else
    x++;
  assert(x != 0);
  int y;
  if (y > 10)
    y--;
  else
    y++;
  assert(y != 11);
  return 0;
}
