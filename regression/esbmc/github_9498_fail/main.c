#include <assert.h>

int main()
{
  int i = 0;
  do
  {
    if (i < 5)
      i++;
  } while (i < 3);
  assert(i == 5);
  return 0;
}
