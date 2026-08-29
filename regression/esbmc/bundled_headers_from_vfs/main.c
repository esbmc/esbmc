#include <string.h>

int main(void)
{
  char b[4];
  strcpy(b, "abc");
  __ESBMC_assert(b[0] == 'a', "strcpy copied the first byte");
  return 0;
}
