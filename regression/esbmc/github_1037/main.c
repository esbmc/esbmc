// Regression for #1037: the interval domain used to print a bogus
// "ESBMC fails to convert constant_floatbv ... value 0.000000 ... into double"
// warning for every zero double constant, because isnormal(0.0) is false.
// Zero converts to 0.0 correctly, so no warning must be emitted.
int main()
{
  double x = 0.0;
  double y = x + 1.0;
  __ESBMC_assert(y > 0.0, "y must be positive");
  return 0;
}
