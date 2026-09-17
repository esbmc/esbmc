#include <assert.h>

int main()
{
  unsigned x = 0;
  for (unsigned i = 0; i < 1000; i++)
  {
    if (x == 3)
      x = 0;
    else
      x++;
    if (i == 997)
      x = 5;
  }
  assert(x < 4);
  return 0;
}
