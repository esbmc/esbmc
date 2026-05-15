#include <string.h>

struct my_bitfields
{
  int a : 8;
  int b : 8;
  int c : 8;
  int d : 8;
};

// memcpy from char[] into a struct with bitfields, verified via char* cast
int main()
{
  char src[50];
  struct my_bitfields dst;
  memcpy(&dst, src, 4);
  char *test = (char *)&dst;
  __ESBMC_assert(src[2] == test[2], "memcpy");
  return 0;
}
