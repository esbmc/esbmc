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
  }
  assert(x < 4);
  return 0;
}
