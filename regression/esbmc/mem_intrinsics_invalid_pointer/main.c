#include <string.h>

int nondet_int();
unsigned nondet_uint();

/* The same operands, read only where they point at an object or for zero
 * bytes, which the builtins' C models never dereference. */
int main()
{
  char a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 4};
  char *p = (char *)0x1000;
  if (nondet_int())
    p = a;

  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= 4);

  if (p == a)
  {
    __ESBMC_assert(memcmp(p, b, 4) == 0, "equal");
    memset(p, 0, 4);
    __ESBMC_assert(memchr(p, 3, 4) == 0, "cleared");
    memcpy(b, p, 4);
    memcpy(p, b, n);
  }

  int r = memcmp(p, b, 0);
  memset(p, 0, 0);
  char *c = memchr(p, 3, 0);
  memcpy(b, p, 0);
  memcpy(p, b, n - n);
  __ESBMC_assert(r == 0 && c == 0, "nothing read");
  return 0;
}
