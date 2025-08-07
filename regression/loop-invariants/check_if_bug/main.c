int nondet_int();

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  int y = nondet_int();
  __ESBMC_assume(y >= 0);
  __ESBMC_assume(y < 1);
  int z = 0;

  __ESBMC_loop_invariant(
    (x >= 0) && (y >= 0) && ((x > y && z == 0) || (x <= y && z == 1)));
  while (x > 0)
  {
    x--;
    if (x == y)
      z = 1; // assert(0);
  }
  assert(z == 0);
}
