extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  int a[4];
  unsigned i = __VERIFIER_nondet_uint();
  unsigned j = __VERIFIER_nondet_uint();
  __ESBMC_assume(i < 4);
  __ESBMC_assume(j < 4);
  __ESBMC_assume(i == j);
  a[i] = 7;
  return a[j];
}
