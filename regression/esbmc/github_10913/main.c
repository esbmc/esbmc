/* esbmc/esbmc#10913: --all-witnesses pushes a frame for its blocking clauses,
   which Boolector rejects unless it was created incremental. */
int nondet_int();
int main()
{
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 10);
  __ESBMC_assert(x != 5, "x != 5");
  return 0;
}
