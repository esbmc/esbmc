#include <string.h>

int nondet_int();
unsigned nondet_uint();

/* An integer-derived address names no object. The builtins resolved a pointer
 * that might also hold one to that object alone and dropped the check. */
int main()
{
  char a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 4};
  char *p = (char *)0x1000;
  if (nondet_int())
    p = a;

  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= 4);

  int r = memcmp(p, b, 4);
  memset(p, 0, 4);
  char *c = memchr(p, 3, 4);
  memcpy(b, p, 4);
  memcpy(p, b, n);
  return r + (c != 0);
}
