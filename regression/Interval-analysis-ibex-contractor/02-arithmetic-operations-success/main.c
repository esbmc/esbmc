#include <assert.h>

int main()
{
  int x = 100;
  float y = 0.1;

  if(y / 5 + 2 * x - 7 < 1)
  {
    x = x + y;
    y = y + 1;
  }
  while(x / 2 - 10 * y < 20)
    x++;

  assert(10 * x > 0);

  return 0;
}