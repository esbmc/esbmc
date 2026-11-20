extern unsigned __VERIFIER_nondet_uint(void);
int main(void)
{
  unsigned x = __VERIFIER_nondet_uint();
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 3 && x + 1 > x);
  while (i < 3)
    i++;
  return 0;
}
