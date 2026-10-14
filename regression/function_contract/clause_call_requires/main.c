int helper(int x) { return x + 1; }

int f(int x)
{
  __ESBMC_requires(helper(x) > 100);
  __ESBMC_ensures(__ESBMC_return_value > 100);
  return x + 1;
}

int main(void)
{
  int n;
  __ESBMC_assume(n > 200);
  return f(n);
}
