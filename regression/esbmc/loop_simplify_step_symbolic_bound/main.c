/* Step recognition handles loops where the bound is a symbol whose
 * value is pinned by an __ESBMC_assume to a singleton interval.
 * Interval analysis folds the bound into a constant_int AND rewrites
 * `!(i < 10)` into `i >= 10`, so the IF guard ends up un-negated.
 * parse_guard accepts both negated and un-negated exit shapes. */
int main()
{
  unsigned N;
  __ESBMC_assume(N == 10);

  int i;
  for (i = 0; i < (int)N; i++)
    ;
  __ESBMC_assert(i == 10, "symbolic bound folded by interval analysis");
  return 0;
}
