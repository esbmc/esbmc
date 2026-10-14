#include <assert.h>

// A store through `(*p)[0]` writes to `arr[0]`, so `arr[0] + b` must be
// recomputed.
int main()
{
  int arr[2];
  arr[0] = 1;
  int b = 2;
  int (*p)[2] = &arr;

  int x = arr[0] + b;
  (*p)[0] = 100;
  int y = arr[0] + b;

  assert(y == 102);
  return 0;
}
