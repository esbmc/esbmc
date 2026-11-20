#include <assert.h>
#include <stdio.h>

int main()
{
  int arr[3] = {1, 2, 3};
  int r = sscanf("5", "%d", &arr[1]);
  if (r == 1)
    assert(arr[1] == 2);
  return 0;
}
