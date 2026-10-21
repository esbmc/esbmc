#include <stdlib.h>
__ESBMC_logic int w(char c) { return c == 'a' ? 2 : 1; }
__ESBMC_logic int len(const char *s, long i)
{
  return i <= 0 ? 0 : len(s, i - 1) + w(s[i - 1]);
}
/* A different definition: not congruent. */
__ESBMC_logic int w_copy(char c) { return c == 'a' ? 2 : 1; }
__ESBMC_logic int len_copy(const char *s, long i)
{
  return i <= 0 ? 0 : len_copy(s, i - 1) + w_copy(s[i - 1]) + 1;
}
long nondet_long();
int main()
{
  long n = nondet_long();
  __ESBMC_assume(n >= 0 && n < 1000000);
  char *s = malloc(n);
  __ESBMC_assert(len(s, n) == len_copy(s, n), "one function");
  free(s);
  return 0;
}
