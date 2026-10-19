/* github #6959: "*" still accepts a function whose clauses call nothing. */
int f(int x)
{
  __ESBMC_requires(x > 200);
  __ESBMC_ensures(__ESBMC_return_value == x + 1);
  return x + 1;
}

int main(void)
{
  int n;
  __ESBMC_assume(n > 200);
  return f(n);
}
