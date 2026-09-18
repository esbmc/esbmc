/* esbmc/esbmc#7901 counterpart: the guard rejecting unknown __ESBMC-prefixed
 * calls must still let the real intrinsics through. */
int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assert(x > 0, "assumed positive");
  return 0;
}
