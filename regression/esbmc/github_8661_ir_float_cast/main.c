// #8661: the SMT lowering of a cast to a signed type under --ir tested
// [0, INT_MAX], so a negative result was reported as an overflow.
double nondet_double();
int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d > -2.0 && d < -1.0);
  int z = (int)d;
  return z;
}
