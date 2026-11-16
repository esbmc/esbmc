int main()
{
  double x = nondet_double();
  __ESBMC_assume(x == 3.0);
  __ESBMC_assert(x / 2.0 == 1.0, "real division");
  return 0;
}
