// Recursion-depth / stack-exhaustion property via function contracts.
//
// Same mechanism as stack_used_bounded, but descend(len) recurses len/2 + 1
// times, keyed to the input. The inductive ensures (proven compositionally) is
// still sound, but the folded subtree's byte peak exceeds the 1,000,000-bit
// budget for a large enough len, so the budget assert FAILS. The recursion is
// never unwound: the contract-replaced call folds it, and the input-keyed
// ensures carries the depth. This is the refutation that flat unwinding cannot
// reach without blowing up.
static void descend(unsigned len)
{
  volatile char buf[1024];
  buf[0] = 0;
  __ESBMC_ensures(
    __ESBMC_stack_used <=
    __ESBMC_old(__ESBMC_stack_used) + (len / 2u + 1u) * __ESBMC_frame_size);
  if (len < 2)
    return;
  descend(len - 2);
}

unsigned nondet_uint(void);
int main(void)
{
  unsigned len = nondet_uint();
  __ESBMC_assume(len <= 65535u);
  descend(len);
  __ESBMC_assert(__ESBMC_stack_used <= 1000000u, "stack budget");
  return 0;
}
