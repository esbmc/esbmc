extern unsigned char __VERIFIER_nondet_uchar(void);
int main(void)
{
  unsigned char c = __VERIFIER_nondet_uchar();
  int i = 0;
  __ESBMC_loop_invariant((int)c >= 0 && i <= 3 && c + 1 > 0);
  while (i < 3)
    i++;
  return 0;
}
