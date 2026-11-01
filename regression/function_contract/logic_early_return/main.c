/* An early return in a definition must not fall through into the next one:
   assuming both results would make every later claim vacuous. */
__ESBMC_logic int sgn(int x)
{
  if (x > 0)
    return 1;
  return 2;
}
int nondet_int();
int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assert(sgn(x) == 2, "sgn(x) is 1 here");
  return 0;
}
