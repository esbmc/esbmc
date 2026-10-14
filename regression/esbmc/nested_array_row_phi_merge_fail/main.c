int nondet_int(void);
int main(void)
{
  int a[2][2];
  int (*p)[2] = a;
  int i = nondet_int();
  int c = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 1;
  a[1][1] = 1;
  if (c)
    a[1][0] = 5;
  else
    a[1][1] = 7;

  __ESBMC_assert(p[1][i] == 1 || p[1][i] == 5,
                 "the else arm's store survives the merge");
  return 0;
}
