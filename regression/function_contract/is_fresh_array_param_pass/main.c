void f(int a[10])
{
  __ESBMC_requires(__ESBMC_is_fresh(a, 10 * sizeof(int)));
}

int main(void)
{
  return 0;
}
