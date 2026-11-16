int main()
{
  float f = nondet_float();
  double d = f;
  __ESBMC_assert(d == f, "widening keeps the value");
  return 0;
}
