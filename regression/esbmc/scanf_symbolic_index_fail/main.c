#include <assert.h>
#include <stdio.h>

int main()
{
  int arr[3] = {1, 2, 3};
  unsigned i;
  __ESBMC_assume(i < 3);
  int r = sscanf("5", "%d", &arr[i]);
  if (r == 1)
    assert(arr[i] == 5);
  return 0;
}
