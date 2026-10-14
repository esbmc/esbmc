// #7974: under --no-propagation the counterexample must come from assume(3 == x).
int main()
{
  int x;
  __ESBMC_assume(3 == x);
  x++;
  __ESBMC_assert(x == 5, "x is 5");
  return 0;
}
