int f(unsigned n)
{
  __ESBMC_requires(n >= 1 && n <= 8);
  __ESBMC_ensures(__ESBMC_return_value == n);
  char buf[n];
  buf[n - 1] = 1;
  return n;
}
