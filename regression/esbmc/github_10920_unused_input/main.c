/* esbmc/esbmc#10920: an input the claim does not read is not enumerated. */
int nondet_int();
int main()
{
  int unused = nondet_int();
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 10);
  __ESBMC_assert(x != 5, "x != 5");
  return 0;
}
