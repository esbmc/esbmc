int main()
{
  unsigned a, b;

  __ESBMC_assume(a < 10);
  __ESBMC_assume(b < 10);
  __ESBMC_assert(a + b < 20, "sum stays bounded");
  __ESBMC_assert(a < 10, "a stays bounded");
  return 0;
}
