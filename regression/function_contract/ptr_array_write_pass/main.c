/* ptr_array_write_pass:
 * Tests contracts when an int* is used as an array (pointer arithmetic).
 * ensures uses arr[0] and arr[1] (i.e. *(arr+0) and *(arr+1)).
 * Caller provides a concrete int[2] — no --assume-nonnull-valid needed.
 */
#include <assert.h>
#include <stddef.h>

void init_pair(int *arr)
{
  __ESBMC_requires(arr != NULL);
  __ESBMC_ensures(arr[0] == 0 && arr[1] == 1);

  arr[0] = 0;
  arr[1] = 1;
}

int main()
{
  int data[2] = {99, 88};
  init_pair(data);
  assert(data[0] == 0 && data[1] == 1);
  return 0;
}
