int nondet_int(void);

int main(void)
{
  int a[2][2];
  int(*p)[2] = a;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  __ESBMC_assert(p[1][i] == 4, "the row's first store is still readable");
  return 0;
}
