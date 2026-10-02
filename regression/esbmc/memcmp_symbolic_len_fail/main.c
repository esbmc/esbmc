#include <string.h>

int main()
{
  char a[4] = "abc";
  char b[4] = "abd";
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 5);
  return memcmp(a, b, n) != 0;
}
