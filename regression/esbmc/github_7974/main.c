// #7974: an assume(x == c) lift left x unconstrained under --no-propagation.
int main()
{
  int var1;
  __ESBMC_assume(var1 == 352005173);
  __ESBMC_assert(var1 != -7, "symbol == constant");

  int x;
  __ESBMC_assume(3 == x);
  x++;
  __ESBMC_assert(x == 4, "constant == symbol");
  return 0;
}
