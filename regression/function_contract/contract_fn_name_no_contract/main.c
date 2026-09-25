int plain(int x)
{
  return x + 1;
}

int f(int x)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  return x > 0 ? x : 0;
}

int main(void)
{
  return f(3) - plain(2);
}
