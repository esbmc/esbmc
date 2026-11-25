/* esbmc/esbmc#10920: a[0] reaches the claim only through the branch guard. */
int nondet_int();
void f(int x)
{
  __ESBMC_assert(x != 0, "x != 0");
}
int main()
{
  int a[2];
  a[0] = nondet_int();
  a[1] = nondet_int();
  if (a[0] >= 0 && a[0] < 3)
    f(a[1]);
  return 0;
}
