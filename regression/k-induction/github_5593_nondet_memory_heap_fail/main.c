// Issue #5593 (recurrence of #5224 / #5230) soundness guard: an UNSAFE program
// whose nondeterministic input is built with __VERIFIER_nondet_memory. Before
// the fix the k-induction inductive step proved the Intel-TDX-Module
// `*_havoc_memory` tasks SAFE because this opaque memory havoc — a write
// through a pointer with no nameable symbol — was not gated (its loop lives in
// a hidden library model, unlike the user nondet-array builders of the
// "havoc_object" shape). Now a reachable call disables the inductive step, so
// the base case finds the genuinely reachable violation: VERIFICATION FAILED,
// never a spurious SUCCESSFUL.
#include <stdlib.h>
extern void __VERIFIER_nondet_memory(void *, __SIZE_TYPE__);

int main(void)
{
  unsigned char *a = malloc(8);
  __ESBMC_assume(a != 0);
  __VERIFIER_nondet_memory(a, 8); // havoc heap memory through a pointer

  __ESBMC_assert(a[7] != 0x2A, "a[7] is nondet, so 0x2A is reachable");
  return 0;
}
