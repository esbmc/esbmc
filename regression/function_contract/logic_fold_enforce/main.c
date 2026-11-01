__ESBMC_logic int w(char c) { return c == 'a' ? 2 : 1; }
__ESBMC_logic int len(const char *s, long i)
{
  return i <= 0 ? 0 : len(s, i - 1) + w(s[i - 1]);
}
int f(const char *s, long n)
{
  __ESBMC_requires(n >= 0 && n <= 1000000 && __ESBMC_is_fresh(s, n));
  __ESBMC_ensures(__ESBMC_return_value == len(s, n));
  int r = 0;
  long i = 0;
  __ESBMC_loop_invariant(0 <= i && i <= n && 0 <= r && r <= 2 * i && r == len(s, i));
  for (; i < n; i++)
    r += s[i] == 'a' ? 2 : 1;
  return r;
}
