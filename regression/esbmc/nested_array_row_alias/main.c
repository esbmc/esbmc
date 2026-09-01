int nondet_int(void);

int main(void)
{
  int a[2][2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  /* Reading the updated row at a nondet index keeps the store chain alive to
     the solver, which is what makes the second write's encoding observable. */
  __ESBMC_assert(a[1][i] == 5 || a[1][i] == 4, "both row stores reach the solver");
  return 0;
}
