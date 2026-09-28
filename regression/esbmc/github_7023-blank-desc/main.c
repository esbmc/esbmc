int main()
{
  int a, b;

  __ESBMC_assert(a + b == b + a, "\t");
  __ESBMC_assert(a == 1, "P2");
  return 0;
}
