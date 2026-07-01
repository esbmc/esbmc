#include <string.h>

// memcpy from char[] to char[], no offset on either side
int main()
{
  char arr[50];
  char arr2[60];
  memcpy(arr, arr2, 50);
  __ESBMC_assert(arr[5] == arr2[5], "memcpy");
  return 0;
}
