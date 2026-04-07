/* ptr_array_write_fail:
 * Body writes arr[0]=0, arr[1]=1 but ensures expects arr[0]==1, arr[1]==0.
 * The swapped expected values must be detected as VERIFICATION FAILED.
 */
#include <stddef.h>

void init_pair(int *arr)
{
  __ESBMC_requires(arr != NULL);
  __ESBMC_ensures(arr[0] == 1 && arr[1] == 0); /* wrong: expected values swapped */

  arr[0] = 0;
  arr[1] = 1;
}

int main()
{
  int data[2] = {99, 88};
  init_pair(data);
  return 0;
}
