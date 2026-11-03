#include <assert.h>

int main()
{
  int i = 0;
  do
  {
    if (i < 5)
      i++;
  } while (i < 3);
  assert(i == 3);

  int j = 0;
  do
  {
    if (j == 4)
      break;
    j++;
  } while (j < 10);
  assert(j == 4);

  int k = 0;
  while (k < 6)
  {
    if (k < 2)
      k++;
    k++;
  }
  assert(k == 6);
  return 0;
}
