// Companion to github_1037: exercises the normal (non-zero) float interval
// path under --interval-analysis and must still report VERIFICATION FAILED
// (the assertion is false), with no spurious "fails to convert" warning.
int main()
{
  double x = 3.0;
  __ESBMC_assert(x < 0.0, "should fail: x is positive");
  return 0;
}
