int main()
{
  int a, b;

  __ESBMC_assert(a + b == b + a, "addition commutes");
  __ESBMC_assert(a * 0 == 0, "zero annihilates");
  return 0;
}
