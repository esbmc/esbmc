int nondet_int(void);
int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m >= 3 && m <= 5);
  int a[2][m][3];
  __ESBMC_assert(sizeof(a) > 72, "declaration only");
  return 0;
}
