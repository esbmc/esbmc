int f(int x)
{
  __ESBMC_ensures(__ESBMC_return_value > x);
  return x + 1;
}

int main(void)
{
  return f(1) - 2;
}
