#include <string.h>

// memcpy from char[] into int[], verified via char* cast of destination
int main()
{
  char src[50];
  int dst[50];
  memcpy(dst, src, 50);
  char *test = (char *)dst;
  __ESBMC_assert(src[42] == test[42], "memcpy");
  return 0;
}
