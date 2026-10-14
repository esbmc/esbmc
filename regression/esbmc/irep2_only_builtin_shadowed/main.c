/* A program may reuse a name-matched spelling; its own definition wins over the
   builtin lowering (#6904). */
double fabs(double x)
{
  return 42.0;
}

double nondet_double(void);

int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d == -3.0);
  __ESBMC_assert(fabs(d) == 42.0, "the program's own fabs is called");
  return 0;
}
