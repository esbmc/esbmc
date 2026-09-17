#include <assert.h>

int main()
{
  int arr[8];
  int *a = arr;
  a[7] = 3;
  unsigned i = 0;
  while (a[i] != 3)
    i++;
  assert(i < 7);
  return 0;
}
