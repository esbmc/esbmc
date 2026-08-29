void f(void)
{
  int a[10];
  __ESBMC_requires(__ESBMC_is_fresh(&a, sizeof(a)));
}

int main(void)
{
  return 0;
}
