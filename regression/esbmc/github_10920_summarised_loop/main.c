/* esbmc/esbmc#10920: the summarised loop makes one array-valued input, which
   has to be blocked element by element. */
int nondet_int();
int main()
{
  int a[2];
  for (int i = 0; i < 2; i++)
    a[i] = nondet_int();
  __ESBMC_assume(a[0] >= 0 && a[0] <= 3 && a[1] >= 0 && a[1] <= 3);
  __ESBMC_assert(a[0] + a[1] != 3, "a[0] + a[1] != 3");
  return 0;
}
