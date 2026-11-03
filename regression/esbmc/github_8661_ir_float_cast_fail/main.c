// The dual of ../github_8661_ir_float_cast: a value below INT_MIN does not
// fit in an int, so the lower bound of a signed cast is still checked.
double nondet_double();
int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d > -3e9 && d < -2.5e9);
  int z = (int)d;
  return z;
}
