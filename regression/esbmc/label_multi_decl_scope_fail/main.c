int main()
{
L:
  int a = 1, b = 2;
  int *p = &b;
  __ESBMC_assert(*p == 1, "b holds 2, not 1");
  return a;
}
