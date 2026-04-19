/* Verifies that ESBMC still catches a real assertion failure when
   an array element is written at the wrong index. */
#include <assert.h>

int main()
{
  int arr[3] = {0, 0, 0};
  arr[1] = 42;
  assert(arr[1] == 0); /* should fail */
  return 0;
}
