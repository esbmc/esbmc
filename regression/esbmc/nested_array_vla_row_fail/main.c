int nondet_int(void);

int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n > 1 && n < 4);
  int a[n][n];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  __ESBMC_assert(a[1][i] == 4, "the symbolic-extent row's earlier store must not be dropped");
  return 0;
}
