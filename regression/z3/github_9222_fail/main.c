// #9222: a signed product below INT_MIN is an overflow.
int nondet_int(void);
int main(void)
{
  int a = nondet_int(), b = nondet_int();
  __ESBMC_assume(a > 0 && b < 0);
  return a * b;
}
