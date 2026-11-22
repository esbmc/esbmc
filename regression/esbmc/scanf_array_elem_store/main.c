#include <assert.h>
#include <stdio.h>

int main()
{
  int arr[3] = {1, 2, 3};
  sscanf("5", "%d", &arr[1]);
  assert(arr[0] == 1);
  assert(arr[2] == 3);
  return 0;
}
