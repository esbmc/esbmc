// --interval-symex-assert without --multi-property: both claims must be
// pruned by the interval domain (0 VCCs remaining) yet no "PASSED (interval)"
// per-claim log line should be emitted.
int main()
{
  int x;
  __ESBMC_assume(x >= -3 && x <= 3);
  __ESBMC_assert(x >= -3, "x lower bound");
  __ESBMC_assert(x <= 3, "x upper bound");
  return 0;
}
