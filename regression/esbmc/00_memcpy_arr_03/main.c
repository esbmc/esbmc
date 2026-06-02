#include <string.h>

// memcpy from char[] to char[], with offset on source pointer
int main()
{
  char arr[50];
  char arr2[50];
  memcpy(arr, arr2 + 10, 40);
  __ESBMC_assert(arr[5] == arr2[15], "memcpy");
  return 0;
}
