// Recursion-depth / stack-exhaustion property via function contracts.
//
// The engine charges the built-in ghost __ESBMC_stack_used by this frame's real
// size (__ESBMC_frame_size) on entry, and havocs it at the contract-replaced
// recursive call, so the inductive ensures bounds the whole folded subtree
// without unwinding the recursion. rec(n) enters n+1 frames; with n <= 4 the
// peak stays under a 1,000,000-bit budget, so this is SAFE.
static void rec(unsigned n)
{
  volatile char buf[1024];
  buf[0] = 0;
  __ESBMC_ensures(
    __ESBMC_stack_used <=
    __ESBMC_old(__ESBMC_stack_used) + (n + 1) * __ESBMC_frame_size);
  if (n == 0)
    return;
  rec(n - 1);
}

unsigned nondet_uint(void);
int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  rec(n);
  __ESBMC_assert(__ESBMC_stack_used <= 1000000u, "stack budget");
  return 0;
}
