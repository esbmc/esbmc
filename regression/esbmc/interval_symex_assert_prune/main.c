// --interval-symex-assert: domain proves both claims from x in [5, 10].
int main()
{
  int x;
  __ESBMC_assume(x >= 5 && x <= 10);
  __ESBMC_assert(x >= 0, "x must be non-negative");
  __ESBMC_assert(x <= 100, "x must be at most 100");
  return 0;
}
