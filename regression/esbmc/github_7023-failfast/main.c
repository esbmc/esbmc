int main()
{
  int a, b, c;

  __ESBMC_assert(a == 1, "P1");
  __ESBMC_assert(b == 2, "P2");
  __ESBMC_assert(c == 3, "P3");
  return 0;
}
