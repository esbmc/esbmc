int helper(int x) { return x + 1; }

int f(int x)
{
  __ESBMC_requires(helper(x) > 100);
  __ESBMC_ensures(__ESBMC_return_value > 100);
  return x + 1;
}

int main(void)
{
  int r = f(500);
  __ESBMC_assert(r > 100, "contract promises this");
  return 0;
}
