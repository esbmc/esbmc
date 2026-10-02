#include <string.h>

int main()
{
  char a[4] = "abc";
  char b[4] = "abd";
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  int r = memcmp(a, b, n);
  __ESBMC_assert(n < 3 ? r == 0 : r < 0, "first difference is at byte 2");
  return 0;
}
