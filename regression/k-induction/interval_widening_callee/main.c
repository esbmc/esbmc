// Widening on entry to check() dropped the bound j < n gives, so j + 1 could
// overflow and j came back unbounded.
void check(int c)
{
  __ESBMC_assert(c, "j >= 0");
}
int main()
{
  int n = nondet_int();
  if (n > 100000)
    return 0;
  for (int j = 0; j < n; j++)
    check(j >= 0);
}
