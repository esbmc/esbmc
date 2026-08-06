void f(void)
{
  int x;
  __ESBMC_requires(__ESBMC_is_fresh(&x, sizeof(int)));
}

int main(void)
{
  return 0;
}
