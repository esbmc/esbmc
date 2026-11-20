extern int __VERIFIER_nondet_int(void);
int main(void)
{
  unsigned x = 0, i = 0;
  if (__VERIFIER_nondet_int())
  {
    for (int t = 0; t < 20; t++)
      x++;
  }
  __ESBMC_loop_invariant(i <= 3 && x == 0);
  while (i < 3)
    i++;
  return 0;
}
