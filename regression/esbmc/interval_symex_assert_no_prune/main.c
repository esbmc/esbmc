// --interval-symex-assert: domain cannot prove x <= 60 from x in [0, 100];
// SMT must still reach the genuine counterexample at x == 100.
int main()
{
  int x;
  __ESBMC_assume(x >= 0 && x <= 100);
  __ESBMC_assert(x <= 60, "x must be at most 60");
  return 0;
}
