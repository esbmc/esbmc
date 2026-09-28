int helper(int x) { return x + 1; }

int f(int x)
{
  __ESBMC_requires(x > 200);
  __ESBMC_ensures(__ESBMC_return_value == helper(x));
  return x + 1;
}

int main(void)
{
  int n;
  __ESBMC_assume(n > 200);
  return f(n);
}
