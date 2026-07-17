int main(void)
{
  /* __builtin_nan("0") folds to a constant_floatbv NaN in the GOTO IR.
   * Asserting n > 0.0 must fail: NaN makes ordered comparisons return false. */
  double n = __builtin_nan("0");

  __ESBMC_assert(n > 0.0, "NaN > 0.0 should not hold");
  return 0;
}
