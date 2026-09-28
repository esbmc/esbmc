int nondet_int(void);
int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m == 715827883); /* 6 * m == 2^32 + 2 */
  int i = nondet_int(), k = nondet_int();
  __ESBMC_assume(i == 1 && k == 2);
  int a[2][2][m][3];
  a[0][0][0][k] = 1;
  a[i][0][0][0] = 2;
  __ESBMC_assert(a[0][0][0][k] == 1, "distinct elements");
  return 0;
}
