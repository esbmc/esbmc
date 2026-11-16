unsigned int __VERIFIER_nondet_uint(void);

static void poll(void)
{
  unsigned char empty = (unsigned char)__VERIFIER_nondet_uint();
  __ESBMC_assume(empty == 1);
  if (!empty)
    poll();
}

int main(void)
{
  unsigned int n = __VERIFIER_nondet_uint();
  for (unsigned int i = 0; i < n; i++)
  {
    poll();
    __ESBMC_assert(i < n, "loop index below bound");
  }
  return 0;
}
