#include <string.h>
int main()
{
  char *p = "abc";
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= 3);
  memcpy(p, "xyz", n);
  return 0;
}
