int main()
{
  int x;

  x = 5;

  __ESBMC_assert(__ESBMC_exists(&x, x == 13), "exists eq");
}
