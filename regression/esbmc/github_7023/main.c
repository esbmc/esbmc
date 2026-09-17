int main()
{
  int a, b, c;

  __ESBMC_assert(a + b == b + a, "A1");
  __ESBMC_assert(a + b == a + c, "A2");
}
