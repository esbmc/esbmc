int nondet_int(void);

int main(void)
{
  int a[2][2];
  int(*p)[2] = a;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  /* Reaching the row through the array symbol rather than its propagated value
     makes the composed store chain the solver's problem: a decomposition that
     kept only the newest update would lose the 5 and invent a counterexample. */
  __ESBMC_assert(p[1][i] == 5 || p[1][i] == 4, "no store the row carries is lost");
  return 0;
}
