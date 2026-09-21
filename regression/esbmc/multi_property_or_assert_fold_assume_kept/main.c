/* The fold dropped the assume(0) after a failed assertion (esbmc/esbmc#7900). */
int main()
{
  int c = nondet_int();
  if (!c)
  {
    __ESBMC_assert(0, "v");
    __ESBMC_assume(0);
  }
  __ESBMC_assert(c != 0, "w");
  return 0;
}
