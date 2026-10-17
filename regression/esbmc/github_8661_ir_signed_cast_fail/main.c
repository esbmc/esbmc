long nondet_long();
int main(void)
{
  long y = nondet_long();
  __ESBMC_assume(y == 2147483648L);
  int x = (int)y;
  return x;
}
