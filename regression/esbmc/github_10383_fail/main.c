// #10383: a VLA with an empty initializer is zero-initialised.
int nondet_int(void);
int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n > 0 && n < 4);
  int a[n] = {};
  __ESBMC_assert(a[n - 1] != 0, "nonzero");
  return 0;
}
