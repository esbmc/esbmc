int inner(int x) { return x; }
int outer(int x) { return x; }

int f(int x)
{
  __ESBMC_requires(outer(inner(x)) > 100);
  __ESBMC_ensures(__ESBMC_return_value > 100);
  return x;
}

int main(void)
{
  int n;
  __ESBMC_assume(n > 200);
  return f(n);
}
