/* esbmc/esbmc#10913: several Boolector re-solves until the blocking clauses
   exhaust the input tuples. */
int nondet_int();
int main()
{
  int x = nondet_int(), y = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 3 && y >= 0 && y <= 3);
  __ESBMC_assert(x + y != 3, "x + y != 3");
  return 0;
}
