/* ptr_sound_array_wrong_index_fail: (soundness)
 * ensures requires arr[0]==7 and arr[1]==8.
 * Body writes arr[1]=7 and arr[0]=8 (indices swapped).
 * Must be VERIFICATION FAILED.
 */
#include <stddef.h>

void fill(int *arr)
{
  __ESBMC_requires(arr != NULL);
  __ESBMC_ensures(arr[0] == 7 && arr[1] == 8);

  arr[1] = 7; /* wrong index */
  arr[0] = 8; /* wrong index */
}

int main()
{
  int data[2] = {0, 0};
  fill(data);
  return 0;
}
