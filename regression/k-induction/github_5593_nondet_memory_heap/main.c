// Issue #5593 (recurrence of #5224 / #5230): __VERIFIER_nondet_memory havocs a
// caller object with fresh nondeterminism through a pointer
// (`*(p + i) = nondet_uchar()` over memory with no nameable symbol). The
// inductive step cannot generalise such an input, so a reachable call to it
// must disable the inductive step — exactly as the per-element user
// nondet-array builder loops do in the SV-COMP "havoc_object" shape. Unlike
// those builders, the write lives in a hidden library model whose body is
// linked into every program, so the gate keys on the *call*, not the loop.
//
// This safe program is proven by the forward condition once the inductive step
// is disabled; the warning confirms the gate fired.
#include <stdlib.h>
extern void __VERIFIER_nondet_memory(void *, __SIZE_TYPE__);

int main(void)
{
  unsigned char *a = malloc(8);
  __ESBMC_assume(a != 0);
  __VERIFIER_nondet_memory(a, 8); // havoc heap memory: disables the inductive step

  unsigned char s = 0;
  for (int i = 0; i < 8; i++)
    s |= a[i]; // bounded loop, fully unwound by the forward condition

  return 0;
}
