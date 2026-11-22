/* esbmc/esbmc#10920: the stores are sliced, so the inputs appear only where
   the claim reads them. */
int nondet_int();
int main()
{
  int a[2];
  a[0] = nondet_int();
  a[1] = nondet_int();
  __ESBMC_assume(a[0] >= 0 && a[0] <= 3 && a[1] >= 0 && a[1] <= 3);
  __ESBMC_assert(a[0] + a[1] != 3, "a[0] + a[1] != 3");
  return 0;
}
