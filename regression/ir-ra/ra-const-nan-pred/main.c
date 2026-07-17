int main(void)
{
  /* __builtin_nan("0") folds to a constant_floatbv NaN in the GOTO IR,
   * exercising the constant-NaN path in convert_terminal under --ir-ieee.
   * All ordered comparisons on NaN return false. */
  double n = __builtin_nan("0");

  __ESBMC_assert(!(n > 0.0), "NaN > 0.0 is false");
  __ESBMC_assert(!(n < 0.0), "NaN < 0.0 is false");
  __ESBMC_assert(!(n == 0.0), "NaN == 0.0 is false");
  return 0;
}
