/* esbmc/esbmc#10920: inputs are numbered in call order, whichever pass finds
   them: y is assigned to a live variable, a[0] and a[1] are read in place. */
int nondet_int();
int main()
{
  int a[2];
  a[0] = nondet_int();
  int y = nondet_int();
  a[1] = nondet_int();
  __ESBMC_assume(y == 10 && a[1] >= 0 && a[1] <= 1 && a[0] >= 5 && a[0] <= 6);
  __ESBMC_assert(a[0] + a[1] + y != 16, "a[0] + a[1] + y != 16");
  return 0;
}
