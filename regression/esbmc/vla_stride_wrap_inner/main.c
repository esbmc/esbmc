int nondet_int(void);
int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m == 1431655766); /* 3 * m == 2^32 + 2 */
  int i = nondet_int(), k = nondet_int();
  __ESBMC_assume(i == 1 && k == 2);
  int a[2][3][m];
  a[0][0][k] = 1;
  a[i][0][0] = 2;
  __ESBMC_assert(a[0][0][k] == 1, "distinct elements");
  return 0;
}
