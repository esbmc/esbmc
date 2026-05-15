#include <string.h>

// memcpy from char[] to char[], with offset on destination pointer
int main()
{
  char arr[50];
  char arr2[50];
  memcpy(arr + 10, arr2, 40);
  __ESBMC_assert(arr[15] == arr2[5], "memcpy");
  return 0;
}
