#include <string.h>

struct my_struct
{
  int a;
  int b;
  int c;
  int d;
  int e;
  char f;
  long *ptr;
};

// memcpy from char[] into a struct, verified via char* cast of destination
int main()
{
  char src[50];
  struct my_struct dst;
  memcpy(&dst, src, 20);
  char *test = (char *)&dst;
  __ESBMC_assert(src[12] == test[12], "memcpy");
  return 0;
}
