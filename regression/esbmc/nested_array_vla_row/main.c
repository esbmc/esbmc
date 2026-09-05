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

  /* A guard, not a gate: the row extent is symbolic, so decompose_stores()
     bails on its constant-size check and the pre-PR encoder runs -- which is
     why this and its _fail twin give the same verdicts on master. What it pins
     is that the new path declines a VLA row rather than mis-flattening one. */
  __ESBMC_assert(a[1][i] == 5 || a[1][i] == 4, "a symbolic-extent row keeps both stores");
  return 0;
}
