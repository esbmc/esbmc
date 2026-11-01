#include <stdlib.h>
__ESBMC_logic int w(char c) { return c == 'a' ? 2 : 1; }
__ESBMC_logic int len(const char *s, long i)
{
  return i <= 0 ? 0 : len(s, i - 1) + w(s[i - 1]);
}
int nondet_int();
int main()
{
  char t[3] = "ab";
  __ESBMC_assert(len(t, 0) == 0 && len(t, 1) == 2, "concrete");
  long n = nondet_int();
  __ESBMC_assume(n >= 0 && n < 1000000);
  char *s = malloc(n + 1);
  int r = 0;
  long i = 0;
  __ESBMC_loop_invariant(0 <= i && i <= n && r == len(s, i));
  for (; i < n; i++)
    r += s[i] == 'a' ? 2 : 1;
  __ESBMC_assert(r == len(s, n), "after loop");
  free(s);
  return 0;
}
