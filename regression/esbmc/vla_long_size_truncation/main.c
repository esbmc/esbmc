int nondet_int(void);
long nondet_long(void);
int main(void)
{
  long u = nondet_long();
  int m = nondet_int();
  __ESBMC_assume(u == 4294967297L && m == 1);
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i <= 1);
  int b[2][u][m];
  b[0][1][0] = 1;
  b[i][0][0] = 2;
  __ESBMC_assert(b[0][1][0] == 1, "b[1][0][0] is not b[0][1][0]");
  return 0;
}
