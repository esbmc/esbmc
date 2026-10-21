#include <stdlib.h>
__ESBMC_logic int w(char c) { return c == 'a' ? 2 : 1; }
__ESBMC_logic int len(const char *s, long i)
{
  return i <= 0 ? 0 : len(s, i - 1) + w(s[i - 1]);
}
int f_old(const char *s, long n)
{
  __ESBMC_requires(n >= 0 && n <= 1000000 && __ESBMC_is_fresh(s, n));
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == len(s, n));
  return 0;
}
int f_new(const char *s, long n)
{
  __ESBMC_requires(n >= 0 && n <= 1000000 && __ESBMC_is_fresh(s, n));
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == len(s, n));
  return 0;
}
long nondet_long();
int main()
{
  long n = nondet_long();
  __ESBMC_assume(n >= 0 && n <= 1000000);
  char *s = malloc(n);
  int a = f_old(s, n);
  int b = f_new(s, n);
  __ESBMC_assert(a == b, "equal");
  free(s);
  return 0;
}
