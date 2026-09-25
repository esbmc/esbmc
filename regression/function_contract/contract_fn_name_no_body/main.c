/* extern_fn is declared, has no body here, and IS called: the flag must be
   rejected because the body is absent, not because nothing calls it. */
int extern_fn(int x);

int f(int x)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  return x > 0 ? x : 0;
}

int main(void)
{
  return f(3) + extern_fn(1) - 3;
}
