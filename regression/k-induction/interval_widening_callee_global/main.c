// Widening on entry to check() must not reach the caller's global after the
// return.
int j;
void check(int c)
{
  __ESBMC_assert(c, "j >= 0");
}
int main()
{
  int n = nondet_int();
  if (n > 100000)
    return 0;
  for (j = 0; j < n; j++)
    check(j >= 0);
}
